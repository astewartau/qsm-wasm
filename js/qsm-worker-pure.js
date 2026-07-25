/**
 * QSM Processing Web Worker - Pure JavaScript/WASM
 *
 * Runs QSM pipeline entirely in WASM without Pyodide.
 * All computation is done in Rust/WASM, no Python dependencies.
 */

// Import utilities - no fallbacks
import { scalePhase, computeB0FromUnwrapped } from './worker/utils/PhaseUtils.js';
import { createThresholdMask, findSeedPoint } from './worker/utils/MaskUtils.js';
import { boxFilter3D, boxFilter3dSeparable } from './worker/utils/FilterUtils.js';
import { computeFieldMap } from './worker/utils/FieldMapping.js';
import { buildConfigJson } from './modules/ConfigBridge.js';
import * as QSMConfig from './app/config.js';

let wasmModule = null;

// Post progress updates to main thread
function postProgress(value, text) {
  self.postMessage({ type: 'progress', value, text });
}

// Post log messages to main thread
function postLog(message) {
  self.postMessage({ type: 'log', message });
}

// Post error to main thread
function postError(message) {
  self.postMessage({ type: 'error', message });
}

// Post completion to main thread
function postComplete(results) {
  self.postMessage({ type: 'complete', results });
}

// Send intermediate stage data for live display
// Set displayNow=false to cache without displaying (e.g., for auxiliary outputs)
function sendStageData(stage, data, dims, voxelSize, affine, description, displayNow = true, displayRange = null) {
  // Save as NIfTI and send to main thread
  const niftiBytes = wasmModule.save_nifti_wasm(
    data,
    dims[0], dims[1], dims[2],
    voxelSize[0], voxelSize[1], voxelSize[2],
    affine
  );
  self.postMessage({ type: 'stageData', stage, data: niftiBytes, description, displayNow, displayRange });
}

function sendStageData4D(stage, echoArrays, dims, voxelSize, affine, description, displayNow = false) {
  // Save first echo as 3D NIfTI, then patch header to 4D and append remaining echoes
  const nEchoes = echoArrays.length;
  const voxelCount = dims[0] * dims[1] * dims[2];

  // Concatenate all echo data
  const allData = new Float64Array(nEchoes * voxelCount);
  for (let e = 0; e < nEchoes; e++) {
    allData.set(echoArrays[e], e * voxelCount);
  }

  // Save as 3D NIfTI (contains all data since save_nifti doesn't check length vs dims)
  const niftiBytes = wasmModule.save_nifti_wasm(
    allData,
    dims[0], dims[1], dims[2],
    voxelSize[0], voxelSize[1], voxelSize[2],
    affine
  );

  // Patch NIfTI header for 4D: dim[0]=4, dim[4]=nEchoes, pixdim[4]=1.0
  // NIfTI-1 header: dim at offset 40 (int16[8]), pixdim at offset 76 (float32[8])
  const header = new DataView(niftiBytes.buffer, niftiBytes.byteOffset, niftiBytes.byteLength);
  header.setInt16(40, 4, true);       // dim[0] = 4 (ndim)
  header.setInt16(48, nEchoes, true); // dim[4] = nEchoes
  header.setFloat32(92, 1.0, true);   // pixdim[4] = 1.0 (needed for NiiVue to show frame controls)

  self.postMessage({ type: 'stageData', stage, data: niftiBytes, description, displayNow });
}

/// Compute a robust display range for non-negative data using percentiles.
/// Apply QSM mean referencing: subtract mean of masked voxels, zero outside mask.
function applyMeanReference(data, mask) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (mask[i]) {
      sum += data[i];
      count++;
    }
  }
  if (count === 0) return data;
  const mean = sum / count;
  const result = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = mask[i] ? (data[i] - mean) : 0;
  }
  return result;
}

function computeRobustRange(data, mask, lowPct = 2, highPct = 98) {
  const values = [];
  for (let i = 0; i < data.length; i++) {
    if ((!mask || mask[i]) && data[i] !== 0 && isFinite(data[i])) {
      values.push(data[i]);
    }
  }
  if (values.length === 0) return null;
  values.sort((a, b) => a - b);
  const lo = values[Math.floor(values.length * lowPct / 100)];
  const hi = values[Math.floor(values.length * highPct / 100)];
  return [lo, hi];
}

async function initializeWasm() {
  try {
    // Construct URLs relative to worker location. Cache-bust by app version so a
    // deployed update fetches fresh WASM rather than a stale cached module (which
    // would throw "<fn> is not a function" when new JS meets old WASM).
    const baseUrl = self.location.href.replace(/\/js\/.*$/, '');
    const v = `?v=${QSMConfig.VERSION}`;
    const jsUrl = `${baseUrl}/wasm/qsm_wasm.js${v}`;
    const wasmBinaryUrl = `${baseUrl}/wasm/qsm_wasm_bg.wasm${v}`;

    const module = await import(jsUrl);
    await module.default(wasmBinaryUrl);
    wasmModule = module;

    if (wasmModule.wasm_health_check()) {
      postLog(`QSMbly v${wasmModule.get_version()} ready`);
    }
  } catch (e) {
    postError(`WASM load failed: ${e.message}`);
    throw e;
  }
}

// Weighted echo fitting - keep inline since it has logging and uses boxFilter3dSeparable
function computeWeightedEchoFit(allUnwrapped, magnitude4d, echoTimes, nx, ny, nz, voxelSize, mask, fitThreshold = 40, fitThreshPercentile = null) {
  const nEchoes = echoTimes.length;
  const voxelCount = nx * ny * nz;
  const teSec = echoTimes.map(t => t / 1000);

  const tfs = new Float64Array(voxelCount);
  const residual = new Float64Array(voxelCount);

  if (nEchoes <= 1) {
    const te = teSec[0];
    const factor = 1 / (2 * Math.PI * te);
    for (let v = 0; v < voxelCount; v++) {
      tfs[v] = mask[v] ? allUnwrapped[v] * factor : 0;
    }
    return { tfs, R_0: new Uint8Array(voxelCount).fill(1) };
  }

  for (let v = 0; v < voxelCount; v++) {
    if (!mask[v]) continue;

    let sumMagPhaseTE = 0;
    let sumMagTESq = 0;

    for (let e = 0; e < nEchoes; e++) {
      const mag = magnitude4d[e][v];
      const phase = allUnwrapped[e * voxelCount + v];
      const te = teSec[e];
      sumMagPhaseTE += mag * phase * te;
      sumMagTESq += mag * te * te;
    }

    const slope = sumMagPhaseTE / (sumMagTESq + 1e-20);
    tfs[v] = slope / (2 * Math.PI);

    let sumMagResidSq = 0;
    let sumMag = 0;
    for (let e = 0; e < nEchoes; e++) {
      const mag = magnitude4d[e][v];
      const phase = allUnwrapped[e * voxelCount + v];
      const predicted = slope * teSec[e];
      const diff = phase - predicted;
      sumMagResidSq += mag * diff * diff;
      sumMag += mag;
    }

    residual[v] = sumMag > 0 ? (sumMagResidSq / sumMag) * nEchoes : 0;
  }

  for (let i = 0; i < voxelCount; i++) {
    if (!isFinite(residual[i])) residual[i] = 0;
  }

  const kx = Math.round(1 / voxelSize[0]) * 2 + 1;
  const ky = Math.round(1 / voxelSize[1]) * 2 + 1;
  const kz = Math.round(1 / voxelSize[2]) * 2 + 1;
  const blurredResidual = boxFilter3dSeparable(residual, nx, ny, nz, kx, ky, kz);

  const nonZeroResiduals = [];
  for (let i = 0; i < voxelCount; i++) {
    if (mask[i] && blurredResidual[i] > 0) nonZeroResiduals.push(blurredResidual[i]);
  }
  nonZeroResiduals.sort((a, b) => a - b);

  if (nonZeroResiduals.length > 0) {
    const minRes = nonZeroResiduals[0];
    const maxRes = nonZeroResiduals[nonZeroResiduals.length - 1];
    const medianRes = nonZeroResiduals[Math.floor(nonZeroResiduals.length / 2)];
    const p90Res = nonZeroResiduals[Math.floor(nonZeroResiduals.length * 0.9)];
    const p99Res = nonZeroResiduals[Math.floor(nonZeroResiduals.length * 0.99)];
    console.log(`[EchoFit] Blurred residual stats: min=${minRes.toFixed(4)}, median=${medianRes.toFixed(4)}, p90=${p90Res.toFixed(4)}, p99=${p99Res.toFixed(4)}, max=${maxRes.toFixed(4)}`);
  }

  let threshold;
  if (fitThreshPercentile !== null) {
    threshold = nonZeroResiduals.length > 0
      ? nonZeroResiduals[Math.min(Math.floor(nonZeroResiduals.length * fitThreshPercentile / 100), nonZeroResiduals.length - 1)]
      : Infinity;
  } else {
    threshold = fitThreshold;
  }
  console.log(`[EchoFit] Using threshold=${threshold.toFixed(4)} (mode: ${fitThreshPercentile !== null ? 'adaptive p' + fitThreshPercentile : 'fixed'})`)

  const R_0 = new Uint8Array(voxelCount);
  for (let i = 0; i < voxelCount; i++) {
    if (mask[i] && blurredResidual[i] < threshold) {
      R_0[i] = 1;
    }
  }

  return { tfs, R_0 };
}

// Shared SWI computation - called from any pipeline after phase unwrapping
function computeSWI(pipelineSettings, unwrappedPhase, magnitude, mask, dims, voxelSize, affine) {
  const [nx, ny, nz] = dims;
  const [vsx, vsy, vsz] = voxelSize;

  postLog('Computing Susceptibility Weighted Image...');

  const swiSettings = pipelineSettings?.swi || { hp_sigma: [4, 4, 0], scaling: 'tanh', strength: 4, mip_window: 7 };
  const scalingMap = { 'tanh': 0, 'negative_tanh': 1, 'positive': 2, 'negative': 3, 'triangular': 4 };
  const scalingType = scalingMap[swiSettings.scaling] || 0;

  const swiResult = new Float64Array(wasmModule.calculate_swi_wasm(
    unwrappedPhase, magnitude, mask,
    nx, ny, nz, vsx, vsy, vsz,
    swiSettings.hp_sigma[0], swiSettings.hp_sigma[1], swiSettings.hp_sigma[2],
    scalingType, swiSettings.strength
  ));

  sendStageData('swi', swiResult, dims, voxelSize, affine, 'SWI');
  postLog('SWI complete');

  // Minimum intensity projection
  const mip_window = swiSettings.mip_window || 0;
  if (mip_window > 0 && mip_window <= nz) {
    const mipResult = new Float64Array(wasmModule.create_mip_wasm(
      swiResult, nx, ny, nz, mip_window
    ));
    const mipNz = nz - mip_window + 1;
    sendStageData('mip', mipResult, [nx, ny, mipNz], voxelSize, affine, 'SWI mIP');
    postLog(`mIP complete (window=${mip_window}, output nz=${mipNz})`);
  }
}

async function runPipeline(data) {
  // Dispatch to the appropriate pipeline based on input mode
  const inputMode = data.inputMode || 'raw';

  if (inputMode === 'totalField' || inputMode === 'localField') {
    const combined_method = data.pipelineSettings?.combined_method || 'none';
    if (combined_method === 'tgv') {
      return await runTgvFieldMapPipeline(data);
    } else if (combined_method === 'qsmart') {
      return await runQsmartFieldMapPipeline(data);
    }
    // Standard pipeline for field map inputs
    if (inputMode === 'totalField') {
      return await runTotalFieldPipeline(data);
    } else {
      return await runLocalFieldPipeline(data);
    }
  }

  // Standard raw pipeline continues below
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // Check for combined method (TGV)
  const combined_method = pipelineSettings?.combined_method || 'none';

  if (combined_method === 'tgv') {
    // Use TGV single-step reconstruction
    return await runTgvPipeline(data);
  } else if (combined_method === 'qsmart') {
    // Use QSMART two-stage reconstruction
    return await runQsmartPipeline(data);
  }

  // Extract pipeline settings needed for logging/dispatch
  const backgroundMethod = pipelineSettings?.bf_algorithm || 'vsharp';
  const dipoleMethod = pipelineSettings?.dipole_inversion || 'rts';
  const mediSettings = pipelineSettings?.medi || { smv: false };

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 10%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');

    const nEchoes = echoTimes.length;
    let magnitude4d = [];
    let phase4d = [];
    let dims, voxelSize, affine;

    for (let e = 0; e < nEchoes; e++) {
      postProgress(0.02 + (e / nEchoes) * 0.08, `Loading echo ${e + 1}/${nEchoes}...`);

      // Load magnitude
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[e]));
      const magData = Array.from(magResult.data);
      dims = Array.from(magResult.dims);
      voxelSize = Array.from(magResult.voxelSize);
      affine = Array.from(magResult.affine);
      magnitude4d.push(magData);

      // Load phase
      const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[e]));
      let phaseData = Array.from(phaseResult.data);

      // Scale phase to [-π, +π] using shared pipeline function
      phaseData = Array.from(wasmModule.scale_phase_to_pi_wasm(new Float64Array(phaseData)));
      phase4d.push(phaseData);

      postLog(`  Echo ${e + 1}: shape ${dims[0]}x${dims[1]}x${dims[2]}`);
    }

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data shape: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // =========================================================================
    // Step 2: Create or load mask (10% - 15%)
    // =========================================================================
    postProgress(0.10, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Loading custom mask...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else {
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      const magSource = hasPreparedMagnitude ? 'prepared' : 'first echo';
      postLog(`Creating threshold mask (${thresholdFraction * 100}%) from ${magSource} magnitude...`);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // =========================================================================
    // Step 3: Field mapping (15% - 45%) — shared with qsmxt.rs
    // =========================================================================
    postProgress(0.15, 'Field mapping...');
    // Canonical TOML from qsmxt-config (serde). Mask is supplied separately to the
    // pipeline, so the config's mask section is irrelevant here ('').
    const configToml = wasmModule.config_json_to_toml_wasm(buildConfigJson(pipelineSettings), '');
    const echoTimesSec = echoTimes.map(t => t / 1000); // ms → seconds

    // Flatten per-echo arrays for WASM
    const phasesFlat = new Float64Array(nEchoes * voxelCount);
    const magsFlat = new Float64Array(nEchoes * voxelCount);
    for (let e = 0; e < nEchoes; e++) {
      phasesFlat.set(phase4d[e], e * voxelCount);
      magsFlat.set(magnitude4d[e], e * voxelCount);
    }

    const fieldResult = wasmModule.run_field_mapping_wasm(
      phasesFlat, magsFlat, mask,
      new Float64Array(echoTimesSec),
      nx, ny, nz, vsx, vsy, vsz,
      magField || 3.0, configToml,
    );

    // Result is [b0_field_ppm..., phase_offset...] or just [b0_field_ppm...]
    let b0Fieldmap = new Float64Array(fieldResult.slice(0, voxelCount));
    const phaseOffset = fieldResult.length > voxelCount
      ? new Float64Array(fieldResult.slice(voxelCount, 2 * voxelCount))
      : null;

    if (phaseOffset) {
      sendStageData('phaseOffset', phaseOffset, dims, voxelSize, affine, 'Phase Offset (rad)', false);
    }

    sendStageData('B0', b0Fieldmap, dims, voxelSize, affine, 'B0 Field Map (ppm)');

    // =========================================================================
    // Step 4: Background field removal (40% - 65%) — shared with qsmxt.rs
    // =========================================================================
    postProgress(0.42, `Background removal (${backgroundMethod})...`);

    const skipBgRemoval = dipoleMethod === 'medi' && mediSettings.smv;
    let localField, erodedMask;

    if (skipBgRemoval) {
      postLog('Background removal: Skipped (MEDI SMV handles it internally)');
      localField = b0Fieldmap;
      erodedMask = mask;
    } else {
      postLog(`Removing background field using ${backgroundMethod.toUpperCase()}...`);
      const bgProgress = (current, total) => {
        postProgress(0.42 + (current / total) * 0.20, `${backgroundMethod.toUpperCase()}: ${current}/${total}`);
      };
      const bgResult = wasmModule.run_bg_removal_wasm(
        b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
        magField || 3.0, configToml, bgProgress,
      );
      localField = new Float64Array(bgResult.slice(0, voxelCount));
      erodedMask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        erodedMask[i] = bgResult[voxelCount + i] > 0.5 ? 1 : 0;
      }
    }

    const erodedCount = erodedMask.reduce((a, b) => a + b, 0);
    postLog(`Eroded mask: ${erodedCount} voxels (${(100 * erodedCount / voxelCount).toFixed(1)}%)`);
    sendStageData('bgRemoved', localField, dims, voxelSize, affine,
      skipBgRemoval ? 'B0 Field (MEDI SMV)' : 'Local Field Map (ppm)');

    // =========================================================================
    // Step 5: Dipole inversion (65% - 95%) — shared with qsmxt.rs
    // =========================================================================
    postProgress(0.67, `Dipole inversion (${dipoleMethod.toUpperCase()})...`);
    postLog(`Running ${dipoleMethod.toUpperCase()} dipole inversion...`);

    // Combine magnitude for MEDI edge weighting
    const magnitudeForInversion = hasPreparedMagnitude
      ? new Float64Array(preparedMagnitude)
      : new Float64Array(magnitude4d[0]);

    const invProgress = (current, total) => {
      postProgress(0.67 + (current / total) * 0.25, `${dipoleMethod.toUpperCase()}: ${current}/${total}`);
    };
    let qsmResult = new Float64Array(wasmModule.run_dipole_inversion_wasm(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      magField || 3.0, new Float64Array(echoTimesSec),
      0, 0, 1, magnitudeForInversion,
      configToml, invProgress,
    ));

    // Already in ppm (pipeline stage handles all unit conversions)
    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (erodedMask[i]) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Apply referencing — shared with qsmxt.rs
    const refMethod = pipelineSettings?.reference_mean === false ? 'none' : 'mean';
    qsmResult = new Float64Array(wasmModule.apply_reference_wasm(qsmResult, erodedMask, refMethod));
    if (refMethod === 'mean') postLog('Applied mean referencing');

    // Send QSM result for display
    postProgress(0.95, 'Sending QSM result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSM Result (ppm)');

    postProgress(1.0, 'Pipeline complete!');
    postLog("Pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// TGV Core — shared reconstruction step for all TGV pipelines
// =========================================================================
async function runTgvCore({
  tgvInputPhase, mask, dims, voxelSize, affine,
  te, fieldstrength, tgvSettings,
  progressStart = 0.40, progressEnd = 0.95, label = 'QSM Result (ppm) - TGV',
  reference_mean = true,
}) {
  const [nx, ny, nz] = dims;
  const [vsx, vsy, vsz] = voxelSize;
  const voxelCount = nx * ny * nz;

  const alphas = wasmModule.tgv_get_default_alpha(tgvSettings.regularization);
  const alpha0 = alphas[0];
  const alpha1 = alphas[1];

  // Use adaptive iterations from WASM if available, otherwise fall back to settings
  const step_size = 3.0;
  let iterations = tgvSettings.iterations;
  if (wasmModule.tgv_get_default_iterations) {
    iterations = wasmModule.tgv_get_default_iterations(vsx, vsy, vsz, step_size);
  }

  postProgress(progressStart, 'Starting TGV reconstruction...');
  postLog(`TGV parameters: alpha0=${alpha0.toFixed(4)}, alpha1=${alpha1.toFixed(4)}, iterations=${iterations}, erosions=${tgvSettings.erosions}`);
  postLog(`Using TE=${(te * 1000).toFixed(2)}ms, B0=${fieldstrength}T`);

  const progressRange = progressEnd - progressStart;
  const tgvProgress = (current, total) => {
    const progress = progressStart + (current / total) * progressRange;
    postProgress(progress, `TGV: Iteration ${current}/${total}`);
  };

  let qsmResult = new Float64Array(wasmModule.tgv_qsm_wasm_with_progress(
    tgvInputPhase, mask, nx, ny, nz, vsx, vsy, vsz,
    0, 0, 1,  // B0 direction
    alpha0, alpha1,
    iterations, tgvSettings.erosions,
    te, fieldstrength,
    tgvProgress
  ));

  let qsmMin = Infinity, qsmMax = -Infinity;
  for (let i = 0; i < voxelCount; i++) {
    if (mask[i] && qsmResult[i] !== 0) {
      if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
      if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
    }
  }
  postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

  // Apply QSM referencing if enabled
  if (reference_mean) {
    postLog('Applying mean referencing...');
    qsmResult = applyMeanReference(qsmResult, mask);
  }

  postProgress(progressEnd, 'Sending QSM result...');
  sendStageData('final', qsmResult, dims, voxelSize, affine, label);

  return qsmResult;
}

// TGV single-step pipeline (raw multi-echo input)
async function runTgvPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  const tgvSettings = pipelineSettings?.tgv || { regularization: 2, iterations: 1000, erosions: 3 };

  // Multi-echo combination settings (same as standard pipeline)
  const unwrapping_algorithm = pipelineSettings?.unwrapping_algorithm || 'romeo';
  const phase_offset_method = pipelineSettings?.phase_offset_method || 'mcpc3ds';
  const b0_estimation = pipelineSettings?.b0_estimation || 'weighted_avg';
  const mcpc3dsSettings = pipelineSettings?.mcpc3ds || { sigma: [10, 10, 5] };
  const b0_weight_type = pipelineSettings?.b0_weight_type || 'phase_snr';
  const linearFitSettings = pipelineSettings?.linearFit || { estimate_offset: true };
  const romeoSettings = pipelineSettings?.romeo || {
    phase_gradient_coherence: true,
    mag_coherence: true,
    mag_weight: true
  };

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 10%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');
    postLog("TGV: Loading data via WASM...");

    const nEchoes = echoTimes.length;
    let magnitude4d = [];
    let phase4d = [];
    let dims, voxelSize, affine;

    for (let e = 0; e < nEchoes; e++) {
      postProgress(0.02 + (e / nEchoes) * 0.08, `Loading echo ${e + 1}/${nEchoes}...`);

      // Load magnitude
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[e]));
      const magData = Array.from(magResult.data);
      dims = Array.from(magResult.dims);
      voxelSize = Array.from(magResult.voxelSize);
      affine = Array.from(magResult.affine);
      magnitude4d.push(magData);

      // Load phase
      const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[e]));
      let phaseData = Array.from(phaseResult.data);

      // Scale phase to [-π, +π]
      phaseData = scalePhase(new Float64Array(phaseData));
      phase4d.push(Array.from(phaseData));

      postLog(`  Echo ${e + 1}: shape ${dims[0]}x${dims[1]}x${dims[2]}`);
    }

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data shape: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // =========================================================================
    // Step 2: Create or load mask (10% - 15%)
    // =========================================================================
    postProgress(0.10, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Loading custom mask...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else {
      // Use prepared magnitude if available (combined/bias-corrected), otherwise first echo
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      const magSource = hasPreparedMagnitude ? 'prepared' : 'first echo';
      postLog(`Creating threshold mask (${thresholdFraction * 100}%) from ${magSource} magnitude...`);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // =========================================================================
    // Step 3: Multi-echo combination (15% - 40%) or single-echo passthrough
    // =========================================================================
    let tgvInputPhase;
    let te;  // Echo time to use for TGV (seconds)
    const fieldstrength = magField || 3.0;

    if (nEchoes > 1) {
      // Multi-echo: field mapping → B0 → convert to phase for TGV
      postLog(`Multi-echo data detected (${nEchoes} echoes), computing B0 field map...`);

      const { b0Fieldmap } = computeFieldMap(wasmModule, {
        phase4d, magnitude4d, echoTimes, mask,
        dims, voxelSize, affine, settings: pipelineSettings,
        postLog, postProgress, sendStageData,
      });

      sendStageData('B0', b0Fieldmap, dims, voxelSize, affine, 'B0 Field Map (Hz)');

      // Convert B0 (Hz) to equivalent phase (radians) for TGV
      te = echoTimes[0] / 1000;
      tgvInputPhase = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        tgvInputPhase[i] = 2 * Math.PI * b0Fieldmap[i] * te;
      }
      postLog(`Converted B0 to equivalent phase using TE=${(te * 1000).toFixed(2)}ms`);

    } else {
      // Single echo: use wrapped phase directly (TGV handles wraps via Laplacian)
      postLog("Single-echo data, using wrapped phase directly for TGV...");
      tgvInputPhase = new Float64Array(phase4d[0]);
      te = echoTimes[0] / 1000;
    }

    // =========================================================================
    // Step 4: TGV reconstruction (40% - 95%)
    // =========================================================================
    await runTgvCore({
      tgvInputPhase, mask, dims, voxelSize, affine,
      te, fieldstrength, tgvSettings,
      progressStart: 0.40, progressEnd: 0.95,
      reference_mean: pipelineSettings?.reference_mean !== false,
    });

    postProgress(1.0, 'TGV pipeline complete!');
    postLog("TGV pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// QSMART Core — shared stages for all QSMART pipelines
// Handles: vasculature detection, two-stage SDF+iLSQR (or iLSQR-only),
// offset adjustment, and ppm scaling.
// =========================================================================
async function runQsmartCore({
  fieldMap,         // Float64Array - total field (Hz) or local field (Hz/ppm)
  mask,             // Uint8Array - brain mask
  R_0,              // Uint8Array - reliability map (all-ones if unavailable)
  magnitudeData,    // Float64Array or null - for vasculature detection
  dims, voxelSize, affine,
  pipelineSettings,
  magField,         // B0 in Tesla
  skipSdf = false,  // true for local field inputs (skip background removal)
  isPpm = false     // true if input is ppm (skip final Hz->ppm conversion)
}) {
  const [nx, ny, nz] = dims;
  const [vsx, vsy, vsz] = voxelSize;
  const voxelCount = nx * ny * nz;

  // QSMART settings with defaults from Demo_QSMART.m
  const qsmartSettings = pipelineSettings?.qsmart || {};
  const sdf_sigma1_stage1 = qsmartSettings.sdf_sigma1_stage1 ?? 10;
  const sdf_sigma2_stage1 = qsmartSettings.sdf_sigma2_stage1 ?? 0;
  const sdf_sigma1_stage2 = qsmartSettings.sdf_sigma1_stage2 ?? 8;
  const sdf_sigma2_stage2 = qsmartSettings.sdf_sigma2_stage2 ?? 2;
  const sdf_spatial_radius = qsmartSettings.sdf_spatial_radius ?? 8;
  const sdf_lower_lim = qsmartSettings.sdf_lower_lim ?? 0.6;
  const sdf_curv_constant = qsmartSettings.sdf_curv_constant ?? 500;
  const useCurvature = qsmartSettings.useCurvature !== false;
  const vasculatureSphereRadiusMm = qsmartSettings.vasc_sphere_radius ?? 8.0;
  const vasculatureSphereRadiusOverride = qsmartSettings.vascSphereRadius ?? qsmartSettings.vasculatureSphereRadius;
  const frangi_scale_min = qsmartSettings.frangi_scale_min ?? 0.5;
  const frangi_scale_max = qsmartSettings.frangi_scale_max ?? 6.0;
  const frangi_scale_ratio = qsmartSettings.frangi_scale_ratio ?? 0.5;
  const frangiScaleMinVoxelOverride = qsmartSettings.frangiScaleRange?.[0] ?? qsmartSettings.frangiScaleMin;
  const frangiScaleMaxVoxelOverride = qsmartSettings.frangiScaleRange?.[1] ?? qsmartSettings.frangiScaleMax;
  const frangiScaleRatioOverride = qsmartSettings.frangiScaleRatio;
  const frangi_c = qsmartSettings.frangi_c ?? 500;
  const ilsqr_tol = qsmartSettings.ilsqr_tol ?? 0.01;
  const ilsqr_max_iter = qsmartSettings.ilsqr_max_iter ?? 50;
  const enableVasculature = qsmartSettings.enableVasculature !== false && magnitudeData !== null;

  const b0Tesla = magField || 7.0;
  const gyro = 2.675e8;
  const ppmFactor = gyro * b0Tesla / 1e6;

  const maskCount = mask.reduce((a, b) => a + b, 0);

  // =========================================================================
  // Vasculature detection
  // =========================================================================
  let vascOnly;
  if (enableVasculature) {
    postProgress(0.20, 'Detecting vasculature (Frangi filter)...');

    const avgVoxelSize = (vsx + vsy + vsz) / 3.0;
    const sphereRadiusVoxels = vasculatureSphereRadiusOverride ?? Math.round(vasculatureSphereRadiusMm / avgVoxelSize);
    const effectiveSphereRadius = Math.max(sphereRadiusVoxels, 2);
    const frangiScaleMin = frangiScaleMinVoxelOverride ?? (frangi_scale_min / avgVoxelSize);
    const frangiScaleMax = frangiScaleMaxVoxelOverride ?? (frangi_scale_max / avgVoxelSize);
    const frangiScaleRatio = frangiScaleRatioOverride ?? (frangi_scale_ratio / avgVoxelSize);
    const effectiveScaleRatio = Math.max(frangiScaleRatio, 0.1);

    postLog(`Generating vasculature mask:`);
    postLog(`  Voxel size: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm (avg=${avgVoxelSize.toFixed(2)}mm)`);
    postLog(`  Sphere radius: ${effectiveSphereRadius} voxels (${vasculatureSphereRadiusMm.toFixed(1)}mm)`);
    postLog(`  Frangi scales: [${frangiScaleMin.toFixed(2)}, ${frangiScaleMax.toFixed(2)}] voxels (step=${effectiveScaleRatio.toFixed(2)})`);
    postLog(`  (Physical: [${frangi_scale_min.toFixed(1)}, ${frangi_scale_max.toFixed(1)}]mm, Frangi C: ${frangi_c})`);

    const vascProgress = (current, total) => {
      postProgress(0.20 + (current / total) * 0.10, `Vasculature: Step ${current}/${total}`);
    };

    vascOnly = new Float64Array(wasmModule.vasculature_mask_wasm_with_progress(
      magnitudeData, mask, nx, ny, nz,
      effectiveSphereRadius,
      frangiScaleMin, frangiScaleMax, effectiveScaleRatio,
      frangi_c,
      vascProgress
    ));

    const vascCount = vascOnly.filter(v => v === 0).length;
    postLog(`Vessel voxels: ${vascCount} (${(100 * vascCount / maskCount).toFixed(1)}% of brain)`);

    const vascDisplay = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      vascDisplay[i] = mask[i] ? (1 - vascOnly[i]) : 0;
    }
    sendStageData('vascDetect', vascDisplay, dims, voxelSize, affine, 'Vessel Detection (Frangi)');
  } else {
    if (!magnitudeData) {
      postLog("No magnitude available - vasculature detection disabled, using full mask for both stages");
    } else {
      postLog("Vasculature detection disabled - using full mask for both stages");
    }
    vascOnly = new Float64Array(voxelCount).fill(1.0);
  }

  // =========================================================================
  // Create weighted mask (mask * R_0)
  // =========================================================================
  const weightedMask = new Float64Array(voxelCount);
  let weightedCount = 0;
  for (let i = 0; i < voxelCount; i++) {
    weightedMask[i] = (mask[i] && R_0[i]) ? 1.0 : 0.0;
    if (weightedMask[i] > 0) weightedCount++;
  }
  postLog(`Weighted mask (mask * R_0): ${weightedCount}/${maskCount} voxels (${(100 * weightedCount / maskCount).toFixed(1)}% of brain)`);

  // =========================================================================
  // Stage 1: SDF (optional) + iLSQR on whole ROI (30% - 50%)
  // =========================================================================
  let lfsStage1;
  if (skipSdf) {
    // Local field input: field map IS the local field, skip SDF
    postLog("Skipping SDF background removal (local field input)");
    lfsStage1 = fieldMap;
  } else {
    postProgress(0.30, 'Stage 1: SDF background removal...');
    postLog(`Stage 1 SDF: sigma1=${sdf_sigma1_stage1}, sigma2=${sdf_sigma2_stage1}, curvature=${useCurvature}`);

    const onesArray = new Float64Array(voxelCount).fill(1.0);
    const sdfProgress1 = (current, total) => {
      postProgress(0.30 + (current / total) * 0.10, `Stage 1 SDF: ${current}/${total} alphas`);
    };

    lfsStage1 = new Float64Array(wasmModule.sdf_wasm_with_progress(
      fieldMap, weightedMask, onesArray,
      nx, ny, nz,
      sdf_sigma1_stage1, sdf_sigma2_stage1,
      sdf_spatial_radius,
      sdf_lower_lim, sdf_curv_constant,
      useCurvature,
      sdfProgress1
    ));

    let lfs1Min = Infinity, lfs1Max = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (weightedMask[i] > 0) {
        if (lfsStage1[i] < lfs1Min) lfs1Min = lfsStage1[i];
        if (lfsStage1[i] > lfs1Max) lfs1Max = lfsStage1[i];
      }
    }
    postLog(`Stage 1 LFS range: [${lfs1Min.toFixed(2)}, ${lfs1Max.toFixed(2)}] ${isPpm ? 'ppm' : 'Hz'}`);
    sendStageData('lfsStage1', lfsStage1, dims, voxelSize, affine, `Stage 1 Local Field (${isPpm ? 'ppm' : 'Hz'})`);
  }

  // Scale local field to ppm for offset adjustment
  const lfsStage1Ppm = new Float64Array(voxelCount);
  if (isPpm) {
    // Already in ppm
    for (let i = 0; i < voxelCount; i++) {
      lfsStage1Ppm[i] = lfsStage1[i];
    }
  } else {
    for (let i = 0; i < voxelCount; i++) {
      lfsStage1Ppm[i] = lfsStage1[i] * ppmFactor;
    }
  }

  // Inner dipole inversion algorithm for both QSMART stages (default iLSQR).
  const innerAlgo = (qsmartSettings.inversion_algorithm || 'ilsqr').toLowerCase();
  // Per-algorithm params come from the QSMART panel's own inputs (qsmartSettings.<algo>);
  // anything unset falls back to the algorithm defaults inside runDipoleInversionByMethod.
  const qsmartInvSettings = {
    ilsqr: { tol: ilsqr_tol, max_iter: ilsqr_max_iter },
    tkd: qsmartSettings.tkd,
    tsvd: qsmartSettings.tsvd,
    tikhonov: qsmartSettings.tikhonov,
    tv: qsmartSettings.tv,
    rts: qsmartSettings.rts,
    nltv: qsmartSettings.nltv,
    medi: qsmartSettings.medi,
  };

  postProgress(0.42, `Stage 1: ${innerAlgo.toUpperCase()} inversion...`);
  postLog(`Stage 1 ${innerAlgo.toUpperCase()} inversion`);

  const maskStage1 = new Uint8Array(voxelCount);
  for (let i = 0; i < voxelCount; i++) {
    maskStage1[i] = weightedMask[i] > 0.1 ? 1 : 0;
  }

  const ilsqrProgress1 = (current, total) => {
    postProgress(0.42 + (current / total) * 0.08, `Stage 1 ${innerAlgo.toUpperCase()}: ${current}/${total}`);
  };

  // QSMART fields are already ppm/Hz local fields; skip the MEDI Hz->rad conversion (no echo times here).
  const chiStage1 = await runDipoleInversionByMethod(
    lfsStage1, maskStage1, nx, ny, nz, vsx, vsy, vsz,
    innerAlgo, qsmartInvSettings,
    magnitudeData, null, isPpm, magField,
    ilsqrProgress1
  );

  let chi1Min = Infinity, chi1Max = -Infinity;
  for (let i = 0; i < voxelCount; i++) {
    if (maskStage1[i]) {
      if (chiStage1[i] < chi1Min) chi1Min = chiStage1[i];
      if (chiStage1[i] > chi1Max) chi1Max = chiStage1[i];
    }
  }
  postLog(`Stage 1 Chi range: [${chi1Min.toFixed(4)}, ${chi1Max.toFixed(4)}]`);
  sendStageData('chiStage1', chiStage1, dims, voxelSize, affine, 'Stage 1 QSM (arb)');

  // =========================================================================
  // Stage 2: SDF (optional) + iLSQR on tissue only (50% - 75%)
  // =========================================================================
  let lfsStage2;
  if (skipSdf) {
    // Local field input: same local field, different mask
    lfsStage2 = fieldMap;
  } else {
    postProgress(0.50, 'Stage 2: SDF on tissue region...');
    postLog(`Stage 2 SDF: sigma1=${sdf_sigma1_stage2}, sigma2=${sdf_sigma2_stage2}`);

    const tfsWeighted = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      tfsWeighted[i] = fieldMap[i] * weightedMask[i];
    }

    const sdfProgress2 = (current, total) => {
      postProgress(0.50 + (current / total) * 0.12, `Stage 2 SDF: ${current}/${total} alphas`);
    };

    lfsStage2 = new Float64Array(wasmModule.sdf_wasm_with_progress(
      tfsWeighted, weightedMask, vascOnly,
      nx, ny, nz,
      sdf_sigma1_stage2, sdf_sigma2_stage2,
      sdf_spatial_radius,
      sdf_lower_lim, sdf_curv_constant,
      useCurvature,
      sdfProgress2
    ));

    sendStageData('lfsStage2', lfsStage2, dims, voxelSize, affine, `Stage 2 Local Field (${isPpm ? 'ppm' : 'Hz'})`);
  }

  postProgress(0.64, `Stage 2: ${innerAlgo.toUpperCase()} inversion...`);

  const maskStage2 = new Uint8Array(voxelCount);
  for (let i = 0; i < voxelCount; i++) {
    maskStage2[i] = (weightedMask[i] > 0.1 && vascOnly[i] > 0.5) ? 1 : 0;
  }

  const ilsqrProgress2 = (current, total) => {
    postProgress(0.64 + (current / total) * 0.10, `Stage 2 ${innerAlgo.toUpperCase()}: ${current}/${total}`);
  };

  const chiStage2 = await runDipoleInversionByMethod(
    lfsStage2, maskStage2, nx, ny, nz, vsx, vsy, vsz,
    innerAlgo, qsmartInvSettings,
    magnitudeData, null, isPpm, magField,
    ilsqrProgress2
  );

  sendStageData('chiStage2', chiStage2, dims, voxelSize, affine, 'Stage 2 QSM (arb)');

  // =========================================================================
  // Combine stages with offset adjustment (75% - 90%)
  // =========================================================================
  postProgress(0.75, 'Combining stages with offset adjustment...');
  postLog("Computing offset adjustment in Fourier space...");

  const removedVoxels = new Float64Array(voxelCount);
  for (let i = 0; i < voxelCount; i++) {
    removedVoxels[i] = weightedMask[i] - vascOnly[i];
  }

  const chiQsmart = new Float64Array(wasmModule.qsmart_adjust_offset_wasm(
    removedVoxels, lfsStage1Ppm, chiStage1, chiStage2,
    nx, ny, nz, vsx, vsy, vsz,
    0, 0, 1,
    ppmFactor
  ));

  // =========================================================================
  // Scale to ppm and finalize (90% - 100%)
  // =========================================================================
  postProgress(0.90, 'Scaling to ppm...');

  // `let`, not `const`: mean referencing below reassigns this.
  let qsmResult = new Float64Array(voxelCount);
  if (!isPpm) {
    const gamma = QSMConfig.PHYSICS.GYROMAGNETIC_RATIO;
    const scaleFactor = 1e6 / (gamma * b0Tesla);
    for (let i = 0; i < voxelCount; i++) {
      qsmResult[i] = chiQsmart[i] * scaleFactor;
      if (!mask[i]) qsmResult[i] = 0;
    }
  } else {
    for (let i = 0; i < voxelCount; i++) {
      qsmResult[i] = chiQsmart[i];
      if (!mask[i]) qsmResult[i] = 0;
    }
  }

  let qsmMin = Infinity, qsmMax = -Infinity;
  for (let i = 0; i < voxelCount; i++) {
    if (mask[i] && qsmResult[i] !== 0) {
      if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
      if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
    }
  }
  postLog(`QSMART QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

  // Apply QSM referencing if enabled
  if (pipelineSettings?.reference_mean !== false) {
    postLog('Applying mean referencing...');
    qsmResult = applyMeanReference(qsmResult, mask);
  }

  postProgress(0.95, 'Sending QSMART result...');
  sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSMART QSM (ppm)');

  return qsmResult;
}

// QSMART two-stage pipeline (raw multi-echo input)
async function runQsmartPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers, echoTimes, magField,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // Settings used in steps 1-3 (remaining QSMART settings parsed in runQsmartCore)
  const qsmartSettings = pipelineSettings?.qsmart || {};
  const fitThreshold = qsmartSettings.fitThreshold ?? 40;
  const fitThreshPercentile = qsmartSettings.fitThreshPercentile ?? null;
  const b0Tesla = magField || 7.0;  // QSMART optimized for 7T

  try {
    // =========================================================================
    // Step 1: Load NIfTI data (0% - 10%)
    // =========================================================================
    postProgress(0.02, 'Loading NIfTI data...');
    postLog("QSMART: Loading multi-echo data...");

    const nEchoes = echoTimes.length;
    let magnitude4d = [];
    let phase4d = [];
    let dims, voxelSize, affine;

    for (let e = 0; e < nEchoes; e++) {
      postProgress(0.02 + (e / nEchoes) * 0.06, `Loading echo ${e + 1}/${nEchoes}...`);

      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[e]));
      magnitude4d.push(Array.from(magResult.data));
      dims = Array.from(magResult.dims);
      voxelSize = Array.from(magResult.voxelSize);
      affine = Array.from(magResult.affine);

      const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[e]));
      let phaseData = scalePhase(new Float64Array(phaseResult.data));
      phase4d.push(Array.from(phaseData));

      postLog(`  Echo ${e + 1}: shape ${dims[0]}x${dims[1]}x${dims[2]}`);
    }

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm, B0=${b0Tesla}T`);

    // =========================================================================
    // Step 2: Create or load mask (10% - 12%)
    // =========================================================================
    postProgress(0.10, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Loading custom mask...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else {
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : new Float64Array(magnitude4d[0]);
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // =========================================================================
    // Step 3: Phase unwrapping and B0 estimation (12% - 20%)
    // =========================================================================
    postProgress(0.12, 'Phase unwrapping (Laplacian)...');
    postLog("Running Laplacian phase unwrapping...");

    // Unwrap first echo
    const phase1 = new Float64Array(phase4d[0]);
    let unwrappedPhase = new Float64Array(wasmModule.laplacian_unwrap_wasm(
      phase1, mask, nx, ny, nz, vsx, vsy, vsz
    ));

    // Multi-echo B0 estimation with magnitude-weighted fitting and R_0
    postProgress(0.16, 'Computing total field shift...');
    let tfs;
    let R_0;
    if (nEchoes > 1) {
      // Unwrap all echoes
      const allUnwrapped = new Float64Array(voxelCount * nEchoes);
      allUnwrapped.set(unwrappedPhase, 0);

      for (let e = 1; e < nEchoes; e++) {
        const phaseE = new Float64Array(phase4d[e]);
        const unwrappedE = new Float64Array(wasmModule.laplacian_unwrap_wasm(
          phaseE, mask, nx, ny, nz, vsx, vsy, vsz
        ));
        allUnwrapped.set(unwrappedE, e * voxelCount);
      }

      // Magnitude-weighted fit + R_0 reliability map (matching echofit.m)
      const fitResult = computeWeightedEchoFit(
        allUnwrapped, magnitude4d, echoTimes, nx, ny, nz, voxelSize, mask, fitThreshold, fitThreshPercentile
      );
      tfs = fitResult.tfs;
      R_0 = fitResult.R_0;

      const r0Count = R_0.reduce((a, b) => a + b, 0);
      const maskCount2 = mask.reduce((a, b) => a + b, 0);
      const excludedCount = maskCount2 - r0Count;
      const threshMode = fitThreshPercentile !== null ? `adaptive percentile=${fitThreshPercentile}` : `fixed=${fitThreshold}`;
      postLog(`R_0 reliability: ${r0Count} reliable voxels, ${excludedCount} excluded (${(100 * excludedCount / maskCount2).toFixed(1)}% of brain, ${threshMode})`);
    } else {
      // Single echo: no residuals to compute R_0 from
      const te = echoTimes[0] / 1000;
      tfs = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        tfs[i] = unwrappedPhase[i] / (2 * Math.PI * te);
      }
      R_0 = new Uint8Array(voxelCount).fill(1);
      postLog("Single echo: R_0 set to all ones (no multi-echo residual available)");
    }

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) tfs[i] = 0;
    }

    // Compute TFS range without spread operator (avoid stack overflow for large arrays)
    let tfsMin = Infinity, tfsMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i]) {
        if (tfs[i] < tfsMin) tfsMin = tfs[i];
        if (tfs[i] > tfsMax) tfsMax = tfs[i];
      }
    }
    postLog(`TFS range: [${tfsMin.toFixed(1)}, ${tfsMax.toFixed(1)}] Hz`);

    // Send TFS as intermediate stage
    sendStageData('tfs', tfs, dims, voxelSize, affine, 'Total Field Shift (Hz)');

    // Send R_0 reliability map for visualization
    if (nEchoes > 1) {
      const r0Float = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        r0Float[i] = R_0[i];
      }
      sendStageData('R0', r0Float, dims, voxelSize, affine, 'Reliability Map R_0', false);
    }

    // =========================================================================
    // Steps 4-9: QSMART core (vasculature, two-stage SDF+iLSQR, combine, ppm)
    // =========================================================================
    const magnitudeForVasc = hasPreparedMagnitude
      ? new Float64Array(preparedMagnitude)
      : new Float64Array(magnitude4d[0]);

    await runQsmartCore({
      fieldMap: tfs,
      mask, R_0,
      magnitudeData: magnitudeForVasc,
      dims, voxelSize, affine,
      pipelineSettings,
      magField: b0Tesla
    });

    postProgress(1.0, 'QSMART pipeline complete!');
    postLog("QSMART two-stage pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// BET-specific handlers
function postBETProgress(value, text) {
  self.postMessage({ type: 'betProgress', value, text });
}

function postBETLog(message) {
  self.postMessage({ type: 'betLog', message });
}

function postBETComplete(maskData, coverage) {
  self.postMessage({ type: 'betComplete', maskData, coverage });
}

function postBETError(message) {
  self.postMessage({ type: 'betError', message });
}

async function runBET(data) {
  const { magnitudeBuffer, fractionalIntensity, smoothnessFactor, gradientThreshold, iterations, subdivisions } = data;
  const betIterations = iterations || 1000;
  const betSubdivisions = subdivisions || 4;
  const betSmoothness = smoothnessFactor ?? 1.0;  // FSL default
  const betGradient = gradientThreshold ?? 0.0;   // FSL default

  try {
    // Load magnitude data
    postBETProgress(0.1, 'Loading data...');
    postBETLog("Loading magnitude image...");

    const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
    const magData = new Float64Array(magResult.data);
    const dims = Array.from(magResult.dims);
    const voxelSize = Array.from(magResult.voxelSize);

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;

    postBETLog(`Image: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // TEST: Create a simple sphere mask to verify data transfer works
    const TEST_SPHERE = false;  // Set to true to test with sphere instead of BET

    let mask;
    if (TEST_SPHERE) {
      postBETProgress(0.2, 'Creating test sphere...');
      postBETLog(`Creating test sphere mask at center (${nx/2}, ${ny/2}, ${nz/2}), radius=${Math.min(nx,ny,nz)/3}`);
      mask = wasmModule.create_sphere_mask(
        nx, ny, nz,
        nx / 2, ny / 2, nz / 2,
        Math.min(nx, ny, nz) / 3
      );
    } else {
      // Run BET with progress callback
      postBETProgress(0.15, 'Running BET...');
      postBETLog(`Running BET (fi=${fractionalIntensity || 0.5}, smooth=${betSmoothness}, grad=${betGradient}, iter=${betIterations}, subdiv=${betSubdivisions})...`);

      // Progress callback that updates the progress bar during iteration
      const progressCallback = (current, total) => {
        // Map iterations to 0.15 - 0.9 range (leave room for mask conversion)
        const progress = 0.15 + (current / total) * 0.75;
        const pct = Math.round((current / total) * 100);
        postBETProgress(progress, `BET iteration ${current}/${total} (${pct}%)`);
      };

      mask = wasmModule.bet_wasm_with_progress(
        magData, nx, ny, nz, vsx, vsy, vsz,
        fractionalIntensity || 0.5, betSmoothness, betGradient,
        betIterations, betSubdivisions,
        progressCallback
      );

      postBETProgress(0.95, 'Converting mask...');
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    const totalVoxels = mask.length;
    const coveragePct = (maskCount / totalVoxels) * 100;

    postBETLog(`Mask coverage: ${maskCount}/${totalVoxels} voxels (${coveragePct.toFixed(1)}%)`);

    // Convert mask to Float32 for transfer
    const maskFloat = new Float32Array(totalVoxels);
    for (let i = 0; i < totalVoxels; i++) {
      maskFloat[i] = mask[i];
    }

    postBETProgress(1.0, 'Complete');
    postBETComplete(maskFloat, `${coveragePct.toFixed(1)}%`);

  } catch (error) {
    postBETError(error.message);
    console.error('BET error:', error);
  }
}

/**
 * Run bias field correction on magnitude data
 * @param {Object} data - Contains magnitude, dimensions, voxel sizes, and parameters
 */
async function runBiasCorrection(data) {
  const { magnitude, nx, ny, nz, vx, vy, vz, sigma_mm, nbox } = data;

  try {
    // Ensure WASM is initialized
    if (!wasmModule) {
      await initializeWasm();
    }

    console.log(`[Worker] Bias correction: ${nx}x${ny}x${nz}, voxel size=${vx.toFixed(2)}x${vy.toFixed(2)}x${vz.toFixed(2)}mm, sigma=${sigma_mm}mm, nbox=${nbox}`);

    const inputArray = new Float64Array(magnitude);
    const inputSum = inputArray.reduce((a, b) => a + b, 0);
    console.log(`[Worker] Input data length: ${inputArray.length}, sum: ${inputSum.toExponential(3)}`);

    // Call WASM bias correction
    const result = wasmModule.makehomogeneous_wasm(
      inputArray,
      nx, ny, nz,
      vx, vy, vz,
      sigma_mm,
      nbox
    );

    const resultSum = result.reduce((a, b) => a + b, 0);
    console.log(`[Worker] Bias correction complete, result sum: ${resultSum.toExponential(3)}`);

    // Send result back
    self.postMessage({
      type: 'biasCorrection',
      result: Array.from(result)
    });

  } catch (error) {
    console.error('[Worker] Bias correction error:', error);
    self.postMessage({
      type: 'biasCorrection',
      error: error.message
    });
  }
}

/**
 * Compute ROMEO voxel quality map for phase-based masking
 * @param {Object} data - Contains phase, mag, phase2, te1, te2, mask, nx, ny, nz
 */
async function runVoxelQuality(data) {
  const { phase, mag, phase2, te1, te2, mask, nx, ny, nz } = data;

  try {
    if (!wasmModule) {
      await initializeWasm();
    }

    console.log(`[Worker] Voxel quality: ${nx}x${ny}x${nz}`);

    // Scale phase to [-π, +π] before quality map computation
    const phaseArray = scalePhase(new Float64Array(phase));
    const magArray = new Float64Array(mag || []);
    const phase2Array = phase2 && phase2.length > 0
      ? scalePhase(new Float64Array(phase2))
      : new Float64Array([]);
    const maskArray = new Uint8Array(mask);

    const result = wasmModule.voxel_quality_romeo_wasm(
      phaseArray, magArray, phase2Array,
      te1 || 1.0, te2 || 1.0,
      maskArray, nx, ny, nz
    );

    console.log(`[Worker] Voxel quality map complete, length: ${result.length}`);

    self.postMessage({
      type: 'voxelQuality',
      result: Array.from(result)
    });

  } catch (error) {
    console.error('[Worker] Voxel quality error:', error);
    self.postMessage({
      type: 'voxelQuality',
      error: error.message
    });
  }
}

// =========================================================================
// Total Field Map Pipeline
// Skips phase unwrapping/combination, starts from B0 field map
// =========================================================================
async function runTotalFieldPipeline(data) {
  const {
    totalFieldBuffer, fieldMapUnits, magnitudeBuffer, maskBuffer,
    customMaskBuffer, magField, maskThreshold, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasMaskFile = maskBuffer !== null && maskBuffer !== undefined;
  const hasMagnitude = magnitudeBuffer !== null && magnitudeBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  // Extract pipeline settings
  const backgroundMethod = pipelineSettings?.bf_algorithm || 'vsharp';
  const dipoleMethod = pipelineSettings?.dipole_inversion || 'rts';

  // Validate methods
  const validBgMethods = ['vsharp', 'sharp', 'resharp', 'ismv', 'pdf', 'lbv', 'harperella', 'iharperella'];
  const validInversionMethods = ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi', 'ndi', 'fansi', 'fansitgv', 'l1qsm', 'whqsm', 'hdqsm', 'ilsqr'];
  if (!validBgMethods.includes(backgroundMethod)) {
    throw new Error(`Unknown background removal method: '${backgroundMethod}'`);
  }
  if (!validInversionMethods.includes(dipoleMethod)) {
    throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'`);
  }

  // Extract method-specific settings (same as standard pipeline)
  const vsharpSettings = {
    max_radius: pipelineSettings?.vsharp?.max_radius ?? 18,
    min_radius: pipelineSettings?.vsharp?.min_radius ?? 2,
    threshold: pipelineSettings?.vsharp?.threshold ?? 0.05
  };
  const lbvSettings = {
    tol: pipelineSettings?.lbv?.tol ?? 0.000001,
    maxit: pipelineSettings?.lbv?.maxit ?? 500
  };
  const resharpSettings = pipelineSettings?.resharp || { radius: 6, tik_reg: 1e-4, tol: 1e-6, max_iter: 30 };
  const harperellaSettings = pipelineSettings?.harperella || { radius: 10, max_iter: 40, tol: 1e-6 };
  const iharperellaSettings = pipelineSettings?.iharperella || { radius: 10, max_iter: 40, tol: 1e-6 };
  const rtsSettings = pipelineSettings?.rts || { delta: 0.15, mu: 100000, rho: 10, max_iter: 20 };
  const tkdSettings = pipelineSettings?.tkd || { threshold: 0.15 };
  const tsvdSettings = pipelineSettings?.tsvd || { threshold: 0.15 };
  const tikhonovSettings = pipelineSettings?.tikhonov || { lambda: 0.01, reg: 'identity' };
  const tvSettings = pipelineSettings?.tv || { lambda: 0.0002, max_iter: 250, tol: 0.001 };
  const nltvSettings = pipelineSettings?.nltv || { lambda: 0.001, mu: 1, max_iter: 250, tol: 0.001, newton_max_iter: 10 };
  const mediSettings = pipelineSettings?.medi || {
    lambda: 7.5e-5, percentage: 0.3, max_iter: 30, cg_max_iter: 10, cg_tol: 0.01, tol: 0.1,
    smv: false, smv_radius: 5, merit: false, data_weighting: 1
  };
  const ilsqrSettings = pipelineSettings?.ilsqr || { tol: 0.01, max_iter: 50 };

  try {
    // =========================================================================
    // Step 1: Load total field map
    // =========================================================================
    postProgress(0.05, 'Loading total field map...');
    postLog("Loading total field map...");

    const fieldResult = wasmModule.load_nifti_wasm(new Uint8Array(totalFieldBuffer));
    let fieldData = new Float64Array(fieldResult.data);
    const dims = Array.from(fieldResult.dims);
    const voxelSize = Array.from(fieldResult.voxelSize);
    const affine = Array.from(fieldResult.affine);
    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Field map shape: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // Load optional magnitude
    // Prefer prepared magnitude (RSS-combined, bias-corrected) over raw file
    let magnitudeData = null;
    if (hasPreparedMagnitude) {
      magnitudeData = new Float64Array(preparedMagnitude);
      postLog("Using prepared magnitude for weighting");
    } else if (hasMagnitude) {
      postProgress(0.08, 'Loading magnitude...');
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
      magnitudeData = new Float64Array(magResult.data);
      postLog("Loaded magnitude image");
    }

    // =========================================================================
    // Step 2: Convert field map units to Hz
    // =========================================================================
    postProgress(0.10, 'Converting field map units...');

    const isPpm = fieldMapUnits === 'ppm';
    if (fieldMapUnits === 'rad_s') {
      // rad/s -> Hz: divide by 2π
      postLog("Converting field map from rad/s to Hz...");
      for (let i = 0; i < voxelCount; i++) {
        fieldData[i] /= (2 * Math.PI);
      }
    } else if (isPpm) {
      // ppm -> Hz: multiply by γ * B0
      // Actually we keep it in ppm and skip the final conversion
      postLog("Field map in ppm - will skip final Hz->ppm conversion");
    } else {
      postLog("Field map already in Hz");
    }

    // =========================================================================
    // Step 3: Load or create mask
    // =========================================================================
    postProgress(0.12, 'Loading mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Using edited mask");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else if (hasMaskFile) {
      postLog("Loading mask from file...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(maskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else if (hasPreparedMagnitude) {
      postLog(`Creating threshold mask from prepared magnitude (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(new Float64Array(preparedMagnitude), thresholdFraction);
    } else if (magnitudeData) {
      postLog(`Creating threshold mask from magnitude (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(magnitudeData, thresholdFraction);
    } else {
      throw new Error("No mask source available. Provide a mask file or magnitude image.");
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Apply mask to field data
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) fieldData[i] = 0;
    }

    // Send B0 for display
    sendStageData('B0', fieldData, dims, voxelSize, affine,
      `Total Field Map (${isPpm ? 'ppm' : 'Hz'})`);

    // =========================================================================
    // Step 4: Background field removal
    // =========================================================================
    // Check if MEDI with SMV is enabled - skip background removal
    const skipBgRemoval = dipoleMethod === 'medi' && mediSettings.smv;
    let localField, erodedMask;

    if (skipBgRemoval) {
      postProgress(0.42, 'Skipping background removal (MEDI SMV handles it)...');
      postLog('Background removal: Skipped - MEDI with SMV enabled');
      localField = fieldData;
      erodedMask = mask;
    } else {
      // Run background removal (reuse same logic as standard pipeline)
      const bgResult = await runBackgroundRemoval(
        fieldData, mask, nx, ny, nz, vsx, vsy, vsz,
        backgroundMethod, pipelineSettings, magField
      );
      localField = bgResult.localField;
      erodedMask = bgResult.erodedMask;
    }

    const erodedCount = erodedMask.reduce((a, b) => a + b, 0);
    postLog(`Eroded mask: ${erodedCount} voxels (${(100 * erodedCount / voxelCount).toFixed(1)}%)`);

    const localFieldLabel = skipBgRemoval ? 'Total Field (MEDI SMV will handle BG removal)' : `Local Field Map (${isPpm ? 'ppm' : 'Hz'})`;
    sendStageData('bgRemoved', localField, dims, voxelSize, affine, localFieldLabel);

    // =========================================================================
    // Step 5: Dipole inversion
    // =========================================================================
    let qsmResult = await runDipoleInversion(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      dipoleMethod, pipelineSettings,
      magnitudeData,
      isPpm ? null : [20], // nominal TE (ms) for MEDI Hz->rad conversion (arbitrary, cancels out); null if ppm
      isPpm, magField
    );

    // Scale to ppm (skip if input was already in ppm)
    if (!isPpm) {
      postProgress(0.92, 'Scaling to ppm...');
      const gamma = QSMConfig.PHYSICS.GYROMAGNETIC_RATIO;
      const b0Tesla = magField || 3.0;
      const scaleFactor = 1e6 / (gamma * b0Tesla);
      for (let i = 0; i < voxelCount; i++) {
        qsmResult[i] *= scaleFactor;
        if (!erodedMask[i]) qsmResult[i] = 0;
      }
    } else {
      for (let i = 0; i < voxelCount; i++) {
        if (!erodedMask[i]) qsmResult[i] = 0;
      }
    }

    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (erodedMask[i]) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Apply QSM referencing if enabled
    if (pipelineSettings?.reference_mean !== false) {
      postLog('Applying mean referencing...');
      qsmResult = applyMeanReference(qsmResult, erodedMask);
    }

    postProgress(0.95, 'Sending QSM result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSM Result (ppm)');

    postProgress(1.0, 'Pipeline complete!');
    postLog("Total field map pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// Local Field Map Pipeline
// Skips unwrapping and background removal, starts from local field
// =========================================================================
async function runLocalFieldPipeline(data) {
  const {
    localFieldBuffer, fieldMapUnits, magnitudeBuffer, maskBuffer,
    customMaskBuffer, magField, maskThreshold, preparedMagnitude, pipelineSettings
  } = data;

  const hasMagnitude = magnitudeBuffer !== null && magnitudeBuffer !== undefined;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasMaskFile = maskBuffer !== null && maskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  const dipoleMethod = pipelineSettings?.dipole_inversion || 'rts';
  const validInversionMethods = ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi', 'ndi', 'fansi', 'fansitgv', 'l1qsm', 'whqsm', 'hdqsm', 'ilsqr'];
  if (!validInversionMethods.includes(dipoleMethod)) {
    throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'`);
  }

  try {
    // =========================================================================
    // Step 1: Load local field map
    // =========================================================================
    postProgress(0.05, 'Loading local field map...');
    postLog("Loading local field map...");

    const fieldResult = wasmModule.load_nifti_wasm(new Uint8Array(localFieldBuffer));
    let localField = new Float64Array(fieldResult.data);
    const dims = Array.from(fieldResult.dims);
    const voxelSize = Array.from(fieldResult.voxelSize);
    const affine = Array.from(fieldResult.affine);
    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Field map shape: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // Load optional magnitude for MEDI/weighting
    // Prefer prepared magnitude (RSS-combined, bias-corrected) over raw file
    let magnitudeData = null;
    if (hasPreparedMagnitude) {
      magnitudeData = new Float64Array(preparedMagnitude);
      postLog("Using prepared magnitude for weighting");
    } else if (hasMagnitude) {
      postProgress(0.08, 'Loading magnitude...');
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
      magnitudeData = new Float64Array(magResult.data);
      postLog("Loaded magnitude image for weighting");
    }

    // =========================================================================
    // Step 2: Convert field map units to Hz
    // =========================================================================
    postProgress(0.10, 'Converting field map units...');

    const isPpm = fieldMapUnits === 'ppm';
    if (fieldMapUnits === 'rad_s') {
      postLog("Converting field map from rad/s to Hz...");
      for (let i = 0; i < voxelCount; i++) {
        localField[i] /= (2 * Math.PI);
      }
    } else if (isPpm) {
      postLog("Field map in ppm - will skip final Hz->ppm conversion");
    } else {
      postLog("Field map already in Hz");
    }

    // =========================================================================
    // Step 3: Load mask
    // =========================================================================
    postProgress(0.15, 'Loading mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Using edited mask");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else if (hasMaskFile) {
      postLog("Loading mask from file...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(maskBuffer));
      const maskData = Array.from(maskResult.data);
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskData[i] > 0.5 ? 1 : 0;
      }
    } else {
      throw new Error("Mask is required for local field map pipeline");
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) localField[i] = 0;
    }

    // Send local field for display
    sendStageData('bgRemoved', localField, dims, voxelSize, affine,
      `Local Field Map (${isPpm ? 'ppm' : 'Hz'})`);

    // =========================================================================
    // Step 4: Dipole inversion
    // =========================================================================
    let qsmResult = await runDipoleInversion(
      localField, mask, nx, ny, nz, vsx, vsy, vsz,
      dipoleMethod, pipelineSettings,
      magnitudeData,
      isPpm ? null : [20], // dummy echo time for MEDI
      isPpm, magField
    );

    // Scale to ppm (skip if input was already in ppm)
    if (!isPpm) {
      postProgress(0.92, 'Scaling to ppm...');
      const gamma = QSMConfig.PHYSICS.GYROMAGNETIC_RATIO;
      const b0Tesla = magField || 3.0;
      const scaleFactor = 1e6 / (gamma * b0Tesla);
      for (let i = 0; i < voxelCount; i++) {
        qsmResult[i] *= scaleFactor;
        if (!mask[i]) qsmResult[i] = 0;
      }
    } else {
      for (let i = 0; i < voxelCount; i++) {
        if (!mask[i]) qsmResult[i] = 0;
      }
    }

    let qsmMin = Infinity, qsmMax = -Infinity;
    for (let i = 0; i < voxelCount; i++) {
      if (mask[i]) {
        if (qsmResult[i] < qsmMin) qsmMin = qsmResult[i];
        if (qsmResult[i] > qsmMax) qsmMax = qsmResult[i];
      }
    }
    postLog(`QSM range: [${qsmMin.toFixed(4)}, ${qsmMax.toFixed(4)}] ppm`);

    // Apply QSM referencing if enabled
    if (pipelineSettings?.reference_mean !== false) {
      postLog('Applying mean referencing...');
      qsmResult = applyMeanReference(qsmResult, mask);
    }

    postProgress(0.95, 'Sending QSM result...');
    sendStageData('final', qsmResult, dims, voxelSize, affine, 'QSM Result (ppm)');

    postProgress(1.0, 'Pipeline complete!');
    postLog("Local field map pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// TGV Field Map Pipeline
// Handles both totalField and localField inputs with TGV reconstruction
// =========================================================================
async function runTgvFieldMapPipeline(data) {
  const {
    totalFieldBuffer, localFieldBuffer, fieldMapUnits,
    magnitudeBuffer, maskBuffer, customMaskBuffer,
    magField, maskThreshold, preparedMagnitude, pipelineSettings
  } = data;

  const inputMode = data.inputMode;
  const isLocalField = inputMode === 'localField';
  const fieldBuffer = isLocalField ? localFieldBuffer : totalFieldBuffer;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasMaskFile = maskBuffer !== null && maskBuffer !== undefined;
  const hasMagnitude = magnitudeBuffer !== null && magnitudeBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  const tgvSettings = pipelineSettings?.tgv || { regularization: 2, iterations: 1000, erosions: 3 };

  try {
    // Load field map
    postProgress(0.05, 'Loading field map...');
    const fieldLabel = isLocalField ? 'local' : 'total';
    postLog(`TGV: Loading ${fieldLabel} field map...`);

    const fieldResult = wasmModule.load_nifti_wasm(new Uint8Array(fieldBuffer));
    let fieldData = new Float64Array(fieldResult.data);
    const dims = Array.from(fieldResult.dims);
    const voxelSize = Array.from(fieldResult.voxelSize);
    const affine = Array.from(fieldResult.affine);
    const [nx, ny, nz] = dims;
    const voxelCount = nx * ny * nz;

    postLog(`Field map shape: ${nx}x${ny}x${nz}, voxel: ${voxelSize[0].toFixed(2)}x${voxelSize[1].toFixed(2)}x${voxelSize[2].toFixed(2)}mm`);

    // Load optional magnitude
    // Prefer prepared magnitude (RSS-combined, bias-corrected) over raw file
    let magnitudeData = null;
    if (hasPreparedMagnitude) {
      magnitudeData = new Float64Array(preparedMagnitude);
      postLog("Using prepared magnitude");
    } else if (hasMagnitude) {
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
      magnitudeData = new Float64Array(magResult.data);
    }

    // Convert units to Hz
    postProgress(0.10, 'Converting field map units...');
    const fieldstrength = magField || 3.0;
    const isPpm = fieldMapUnits === 'ppm';

    if (fieldMapUnits === 'rad_s') {
      postLog("Converting field map from rad/s to Hz...");
      for (let i = 0; i < voxelCount; i++) {
        fieldData[i] /= (2 * Math.PI);
      }
    } else if (isPpm) {
      // Convert ppm to Hz for phase conversion: Hz = ppm * γ * B0 / 1e6
      postLog("Converting field map from ppm to Hz for TGV...");
      const gamma = QSMConfig.PHYSICS.GYROMAGNETIC_RATIO;
      const ppmToHz = gamma * fieldstrength / 1e6;
      for (let i = 0; i < voxelCount; i++) {
        fieldData[i] *= ppmToHz;
      }
    } else {
      postLog("Field map already in Hz");
    }

    // Load or create mask
    postProgress(0.12, 'Loading mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Using edited mask");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else if (hasMaskFile) {
      postLog("Loading mask from file...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(maskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else if (hasPreparedMagnitude) {
      postLog(`Creating threshold mask from prepared magnitude (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(new Float64Array(preparedMagnitude), thresholdFraction);
    } else if (magnitudeData) {
      postLog(`Creating threshold mask from magnitude (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(magnitudeData, thresholdFraction);
    } else {
      throw new Error("No mask source available. Provide a mask file or magnitude image.");
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) fieldData[i] = 0;
    }

    // Send field map for display
    sendStageData('B0', fieldData, dims, voxelSize, affine,
      `${isLocalField ? 'Local' : 'Total'} Field Map (Hz)`);

    // Convert Hz to phase for TGV: phase = 2π × f_Hz × TE
    const te = 0.020;  // Nominal 20ms TE (arbitrary, TGV uses TE+B0 for correct scaling)
    const tgvInputPhase = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      tgvInputPhase[i] = 2 * Math.PI * fieldData[i] * te;
    }
    postLog(`Converted field map to phase using nominal TE=${(te * 1000).toFixed(1)}ms`);

    // Run TGV reconstruction
    await runTgvCore({
      tgvInputPhase, mask, dims, voxelSize, affine,
      te, fieldstrength, tgvSettings,
      progressStart: 0.15, progressEnd: 0.95,
      label: `QSM Result (ppm) - TGV (from ${fieldLabel} field)`,
      reference_mean: pipelineSettings?.reference_mean !== false,
    });

    postProgress(1.0, 'TGV pipeline complete!');
    postLog(`TGV ${fieldLabel} field pipeline completed successfully!`);
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// QSMART Field Map Pipeline
// Handles both totalField and localField inputs with QSMART reconstruction
// =========================================================================
async function runQsmartFieldMapPipeline(data) {
  const {
    totalFieldBuffer, localFieldBuffer, fieldMapUnits,
    magnitudeBuffer, maskBuffer, customMaskBuffer,
    magField, maskThreshold, preparedMagnitude, pipelineSettings
  } = data;

  const inputMode = data.inputMode;
  const isLocalField = inputMode === 'localField';
  const fieldBuffer = isLocalField ? localFieldBuffer : totalFieldBuffer;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasMaskFile = maskBuffer !== null && maskBuffer !== undefined;
  const hasMagnitude = magnitudeBuffer !== null && magnitudeBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  try {
    // Load field map
    postProgress(0.05, 'Loading field map...');
    const fieldLabel = isLocalField ? 'local' : 'total';
    postLog(`QSMART: Loading ${fieldLabel} field map...`);

    const fieldResult = wasmModule.load_nifti_wasm(new Uint8Array(fieldBuffer));
    let fieldData = new Float64Array(fieldResult.data);
    const dims = Array.from(fieldResult.dims);
    const voxelSize = Array.from(fieldResult.voxelSize);
    const affine = Array.from(fieldResult.affine);
    const [nx, ny, nz] = dims;
    const voxelCount = nx * ny * nz;
    const b0Tesla = magField || 7.0;

    postLog(`Field map: ${nx}x${ny}x${nz}, voxel: ${voxelSize[0].toFixed(2)}x${voxelSize[1].toFixed(2)}x${voxelSize[2].toFixed(2)}mm, B0=${b0Tesla}T`);

    // Load optional magnitude (for vasculature detection)
    // Prefer prepared magnitude (RSS-combined, bias-corrected) over raw file
    let magnitudeData = null;
    if (hasPreparedMagnitude) {
      magnitudeData = new Float64Array(preparedMagnitude);
      postLog("Using prepared magnitude for vasculature detection");
    } else if (hasMagnitude) {
      postProgress(0.08, 'Loading magnitude...');
      const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffer));
      magnitudeData = new Float64Array(magResult.data);
      postLog("Loaded magnitude image for vasculature detection");
    }

    // Convert units to Hz
    postProgress(0.10, 'Converting field map units...');
    const isPpm = fieldMapUnits === 'ppm';

    if (fieldMapUnits === 'rad_s') {
      postLog("Converting field map from rad/s to Hz...");
      for (let i = 0; i < voxelCount; i++) {
        fieldData[i] /= (2 * Math.PI);
      }
    } else if (isPpm) {
      postLog("Field map in ppm - will skip final Hz->ppm conversion");
    } else {
      postLog("Field map already in Hz");
    }

    // Load or create mask
    postProgress(0.12, 'Loading mask...');
    let mask;

    if (hasCustomMask) {
      postLog("Using edited mask");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else if (hasMaskFile) {
      postLog("Loading mask from file...");
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(maskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else if (magnitudeData) {
      postLog(`Creating threshold mask from magnitude (${thresholdFraction * 100}%)...`);
      mask = createThresholdMask(magnitudeData, thresholdFraction);
    } else {
      throw new Error("Mask is required for QSMART. Provide a mask file or magnitude image.");
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask coverage: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Apply mask
    for (let i = 0; i < voxelCount; i++) {
      if (!mask[i]) fieldData[i] = 0;
    }

    // Send field map for display
    sendStageData(isLocalField ? 'bgRemoved' : 'tfs', fieldData, dims, voxelSize, affine,
      `${isLocalField ? 'Local' : 'Total'} Field Map (${isPpm ? 'ppm' : 'Hz'})`);

    // R_0 = all-ones (no multi-echo data for reliability estimation)
    const R_0 = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      R_0[i] = mask[i];
    }
    postLog("R_0 set to mask (no multi-echo data for reliability estimation)");

    // Run QSMART core
    await runQsmartCore({
      fieldMap: fieldData,
      mask, R_0,
      magnitudeData,
      dims, voxelSize, affine,
      pipelineSettings,
      magField: b0Tesla,
      skipSdf: isLocalField,
      isPpm
    });

    postProgress(1.0, 'QSMART pipeline complete!');
    postLog(`QSMART ${fieldLabel} field pipeline completed successfully!`);
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// =========================================================================
// Shared helper: Background field removal
// Extracted from runPipeline to reuse in totalField mode
// =========================================================================
async function runBackgroundRemoval(
  b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
  backgroundMethod, pipelineSettings, magField
) {
  const voxelCount = nx * ny * nz;
  const vsharpSettings = {
    max_radius: pipelineSettings?.vsharp?.max_radius ?? 18,
    min_radius: pipelineSettings?.vsharp?.min_radius ?? 2,
    threshold: pipelineSettings?.vsharp?.threshold ?? 0.05
  };
  let localField, erodedMask;

  if (backgroundMethod === 'vsharp') {
    postProgress(0.42, 'Preparing V-SHARP background removal...');
    postLog(`Removing background field using V-SHARP...`);
    const radii = [];
    for (let r = vsharpSettings.max_radius; r >= vsharpSettings.min_radius; r -= 2) {
      radii.push(r);
    }
    postLog(`  V-SHARP radii: ${radii.map(r => r.toFixed(1)).join(', ')}`);
    const vsharpProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `V-SHARP: Radius ${current}/${total}`);
    };
    const result = wasmModule.vsharp_wasm_with_progress(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      new Float64Array(radii), vsharpSettings.threshold,
      magField || 3.0, vsharpProgress
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else if (backgroundMethod === 'pdf') {
    postProgress(0.42, 'Preparing PDF background removal...');
    postLog(`Removing background field using PDF...`);
    const pdfSettings = pipelineSettings?.pdf || { tol: 0.00001, maxit: 100 };
    const pdfProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `PDF: Iteration ${current}/${total}`);
    };
    localField = new Float64Array(wasmModule.pdf_wasm_with_progress(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, pdfSettings.tol, pdfSettings.maxit,
      magField || 3.0, pdfProgress
    ));
    erodedMask = mask;
  } else if (backgroundMethod === 'ismv') {
    postProgress(0.42, 'Preparing iSMV background removal...');
    postLog(`Removing background field using iSMV...`);
    const ismvSettings = pipelineSettings?.ismv || { radius: 5, tol: 0.001, maxit: 500 };
    // Compute default radius from voxel size if not set (matches QSM.jl: 2 * max(vsz))
    if (ismvSettings.radius == null || isNaN(ismvSettings.radius) || ismvSettings.radius <= 0) {
      ismvSettings.radius = Math.round(Math.max(2, 2 * Math.max(vsx, vsy, vsz)));
      postLog(`  iSMV: computed default radius=${ismvSettings.radius}mm from voxel size`);
    }
    postLog(`  iSMV params: radius=${ismvSettings.radius}, tol=${ismvSettings.tol}, maxit=${ismvSettings.maxit}`);
    const ismvProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `iSMV: Iteration ${current}/${total}`);
    };
    const result = wasmModule.ismv_wasm_with_progress(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      ismvSettings.radius, ismvSettings.tol, ismvSettings.maxit,
      magField || 3.0, ismvProgress
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else if (backgroundMethod === 'sharp') {
    postProgress(0.42, 'Preparing SHARP background removal...');
    postLog(`Removing background field using SHARP...`);
    const sharpSettings = pipelineSettings?.sharp || { radius: 6, threshold: 0.05 };
    postProgress(0.45, `SHARP: Processing radius ${sharpSettings.radius}mm...`);
    const result = wasmModule.sharp_wasm(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      sharpSettings.radius, sharpSettings.threshold,
      magField || 3.0
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else if (backgroundMethod === 'lbv') {
    postProgress(0.42, 'Preparing LBV background removal...');
    postLog(`Removing background field using LBV...`);
    const lbvSettings = {
    tol: pipelineSettings?.lbv?.tol ?? 0.000001,
    maxit: pipelineSettings?.lbv?.maxit ?? 500
  };
    const lbvProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `LBV: Iteration ${current}/${total}`);
    };
    const result = wasmModule.lbv_wasm_with_progress(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      lbvSettings.tol, lbvSettings.maxit,
      magField || 3.0, lbvProgress
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else if (backgroundMethod === 'resharp') {
    postProgress(0.42, 'Preparing RESHARP background removal...');
    postLog('Removing background field using RESHARP...');
    const resharpSettings = pipelineSettings?.resharp || { radius: 6, tik_reg: 1e-4, tol: 1e-6, max_iter: 30 };
    postLog(`  radius=${resharpSettings.radius}mm, tik_reg=${resharpSettings.tik_reg}, max_iter=${resharpSettings.max_iter}`);
    const resharpProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `RESHARP: Iteration ${current}/${total}`);
    };
    const result = wasmModule.resharp_wasm_with_progress(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      resharpSettings.radius, resharpSettings.tik_reg, resharpSettings.tol, resharpSettings.max_iter,
      magField || 3.0, resharpProgress
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else if (backgroundMethod === 'harperella' || backgroundMethod === 'iharperella') {
    const label = backgroundMethod === 'iharperella' ? 'iHARPERELLA' : 'HARPERELLA';
    const settings = backgroundMethod === 'iharperella'
      ? (pipelineSettings?.iharperella || { radius: 10, max_iter: 40, tol: 1e-6 })
      : (pipelineSettings?.harperella || { radius: 10, max_iter: 40, tol: 1e-6 });
    postProgress(0.42, `Preparing ${label} background removal...`);
    postLog(`Removing background field using ${label}...`);
    postLog(`  radius=${settings.radius}mm, max_iter=${settings.max_iter}`);
    const harpProgress = (current, total) => {
      postProgress(0.42 + (current / total) * 0.20, `${label}: Iteration ${current}/${total}`);
    };
    const wasm_fn = backgroundMethod === 'iharperella'
      ? wasmModule.iharperella_wasm_with_progress
      : wasmModule.harperella_wasm_with_progress;
    const result = wasm_fn(
      b0Fieldmap, mask, nx, ny, nz, vsx, vsy, vsz,
      settings.radius, settings.max_iter, settings.tol,
      harpProgress
    );
    localField = new Float64Array(result.slice(0, voxelCount));
    erodedMask = new Uint8Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      erodedMask[i] = result[voxelCount + i] > 0.5 ? 1 : 0;
    }
  } else {
    throw new Error(`Unknown background removal method: '${backgroundMethod}'`);
  }

  return { localField, erodedMask };
}

// =========================================================================
// Shared helper: Dipole inversion
// Extracted from runPipeline to reuse in field map modes
// =========================================================================
// Dispatch a single dipole inversion to the matching WASM binding.
// `progressCb(current, total)` reports per-iteration progress; the caller owns the
// absolute progress band, so this helper is reusable by both the standard pipeline
// and QSMART's two inner inversion stages.
async function runDipoleInversionByMethod(
  localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
  dipoleMethod, pipelineSettings,
  magnitudeData, echoTimes, skipHzConversion, magField,
  progressCb
) {
  const voxelCount = nx * ny * nz;
  const progress = typeof progressCb === 'function' ? progressCb : () => {};
  const rtsSettings = pipelineSettings?.rts || { delta: 0.15, mu: 100000, rho: 10, max_iter: 20 };
  const tkdSettings = pipelineSettings?.tkd || { threshold: 0.15 };
  const tsvdSettings = pipelineSettings?.tsvd || { threshold: 0.15 };
  const tikhonovSettings = pipelineSettings?.tikhonov || { lambda: 0.01, reg: 'identity' };
  const tvSettings = pipelineSettings?.tv || { lambda: 0.0002, max_iter: 250, tol: 0.001 };
  const nltvSettings = pipelineSettings?.nltv || { lambda: 0.001, mu: 1, max_iter: 250, tol: 0.001, newton_max_iter: 10 };
  const mediSettings = pipelineSettings?.medi || {
    lambda: 7.5e-5, percentage: 0.3, max_iter: 30, cg_max_iter: 10, cg_tol: 0.01, tol: 0.1,
    smv: false, smv_radius: 5, merit: false, data_weighting: 1
  };
  const ilsqrSettings = pipelineSettings?.ilsqr || { tol: 0.01, max_iter: 50 };
  const ndiSettings = pipelineSettings?.ndi || { tau: 2, alpha: 1e-5, max_iter: 200 };
  const fansiSettings = pipelineSettings?.fansi || {
    alpha1: 0.0002, mu1: 0.02, mu2: 1, alpha0: 0.0004, mu0: 0.04, max_iter: 150, tol_update: 0.1
  };
  const fansitgvSettings = pipelineSettings?.fansitgv || pipelineSettings?.fansi || {
    alpha1: 0.0002, mu1: 0.02, mu2: 1, alpha0: 0.0004, mu0: 0.04, max_iter: 150, tol_update: 0.1
  };
  const l1qsmSettings = pipelineSettings?.l1qsm || {
    alpha1: 0.0002, mu1: 0.02, mu2: 1, mu3: 1, lambda: 1, max_iter: 50, tol_update: 1
  };
  const whqsmSettings = pipelineSettings?.whqsm || {
    alpha1: 0.0002, mu1: 0.02, mu2: 1, beta: 150, muh: 3, max_iter: 300, tol_update: 0.1
  };
  const hdqsmSettings = pipelineSettings?.hdqsm || {
    alpha_l2: 0.0001, mu1_l2: 0.01, mu2: 1, max_iter_l1: 20, max_iter_l2: 80, tol_update: 1
  };

  let qsmResult;

  if (dipoleMethod === 'tkd') {
    qsmResult = new Float64Array(wasmModule.tkd_wasm(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, tkdSettings.threshold,
      magField || 3.0
    ));
    progress(1, 1);
  } else if (dipoleMethod === 'tsvd') {
    qsmResult = new Float64Array(wasmModule.tsvd_wasm(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, tsvdSettings.threshold,
      magField || 3.0
    ));
    progress(1, 1);
  } else if (dipoleMethod === 'tikhonov') {
    const regType = { 'identity': 0, 'gradient': 1, 'laplacian': 2 }[tikhonovSettings.reg] || 0;
    qsmResult = new Float64Array(wasmModule.tikhonov_wasm(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, tikhonovSettings.lambda, regType,
      magField || 3.0
    ));
    progress(1, 1);
  } else if (dipoleMethod === 'tv') {
    const rho = tvSettings.rho || 100 * tvSettings.lambda;
    qsmResult = new Float64Array(wasmModule.tv_admm_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, tvSettings.lambda, rho, tvSettings.tol, tvSettings.max_iter,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'rts') {
    qsmResult = new Float64Array(wasmModule.rts_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      rtsSettings.delta, rtsSettings.mu, rtsSettings.rho,
      0.01, rtsSettings.max_iter, 4,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'nltv') {
    qsmResult = new Float64Array(wasmModule.nltv_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      nltvSettings.lambda, nltvSettings.mu,
      nltvSettings.tol, nltvSettings.max_iter, nltvSettings.newton_max_iter,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'medi') {
    // MEDI requires magnitude for gradient weighting
    const magData = magnitudeData || new Float64Array(voxelCount).fill(1.0);
    if (!magnitudeData) {
      postLog("MEDI: No magnitude available, using uniform weighting");
    }
    if (mediSettings.smv) {
      postLog(`MEDI SMV preprocessing enabled: radius=${mediSettings.smv_radius}mm`);
    }

    const nStd = new Float64Array(voxelCount).fill(1.0);

    // Convert local field from Hz to radians for MEDI (unless already ppm)
    let localFieldForMedi = localField;
    let hzToRad = 1.0;
    if (!skipHzConversion && echoTimes && echoTimes.length > 0) {
      const te1Sec = echoTimes[0] / 1000;
      hzToRad = 2 * Math.PI * te1Sec;
      localFieldForMedi = new Float64Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        localFieldForMedi[i] = localField[i] * hzToRad;
      }
      postLog(`MEDI: Converting local field to radians (TE1=${(te1Sec * 1000).toFixed(2)}ms)`);
    } else if (skipHzConversion) {
      postLog("MEDI: Field map in ppm - using values directly (no Hz-to-rad conversion)");
    }

    qsmResult = new Float64Array(wasmModule.medi_l1_wasm_with_progress(
      localFieldForMedi, nStd, magData, erodedMask,
      nx, ny, nz, vsx, vsy, vsz, 0, 0, 1,
      mediSettings.lambda, mediSettings.merit, mediSettings.smv, mediSettings.smv_radius,
      mediSettings.data_weighting, mediSettings.percentage,
      mediSettings.cg_tol, mediSettings.cg_max_iter, mediSettings.max_iter, mediSettings.tol,
      progress
    ));

    // Convert back from radians if needed
    if (!skipHzConversion && hzToRad !== 1.0) {
      const radToHz = 1.0 / hzToRad;
      for (let i = 0; i < voxelCount; i++) {
        qsmResult[i] *= radToHz;
      }
    }
  } else if (dipoleMethod === 'ndi') {
    qsmResult = new Float64Array(wasmModule.ndi_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      ndiSettings.tau, ndiSettings.alpha, ndiSettings.max_iter,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'fansi') {
    qsmResult = new Float64Array(wasmModule.fansi_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      fansiSettings.alpha1, fansiSettings.mu1, fansiSettings.mu2,
      fansiSettings.alpha0, fansiSettings.mu0,
      fansiSettings.max_iter, fansiSettings.tol_update, false,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'fansitgv') {
    qsmResult = new Float64Array(wasmModule.fansi_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      fansitgvSettings.alpha1, fansitgvSettings.mu1, fansitgvSettings.mu2,
      fansitgvSettings.alpha0, fansitgvSettings.mu0,
      fansitgvSettings.max_iter, fansitgvSettings.tol_update, true,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'l1qsm') {
    qsmResult = new Float64Array(wasmModule.l1qsm_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      l1qsmSettings.alpha1, l1qsmSettings.mu1, l1qsmSettings.mu2,
      l1qsmSettings.mu3, l1qsmSettings.lambda,
      l1qsmSettings.max_iter, l1qsmSettings.tol_update,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'whqsm') {
    qsmResult = new Float64Array(wasmModule.whqsm_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      whqsmSettings.alpha1, whqsmSettings.mu1, whqsmSettings.mu2,
      whqsmSettings.beta, whqsmSettings.muh,
      whqsmSettings.max_iter, whqsmSettings.tol_update,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'hdqsm') {
    qsmResult = new Float64Array(wasmModule.hdqsm_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1,
      hdqsmSettings.alpha_l2, hdqsmSettings.mu1_l2, hdqsmSettings.mu2,
      hdqsmSettings.max_iter_l1, hdqsmSettings.max_iter_l2, hdqsmSettings.tol_update,
      magField || 3.0, progress
    ));
  } else if (dipoleMethod === 'ilsqr') {
    qsmResult = new Float64Array(wasmModule.ilsqr_wasm_with_progress(
      localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
      0, 0, 1, ilsqrSettings.tol, ilsqrSettings.max_iter,
      magField || 3.0, progress
    ));
  } else {
    throw new Error(`Unknown dipole inversion method: '${dipoleMethod}'`);
  }

  return qsmResult;
}

// Standard (non-QSMART) dipole inversion: owns the 0.67–0.92 progress band.
async function runDipoleInversion(
  localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
  dipoleMethod, pipelineSettings,
  magnitudeData, echoTimes, skipHzConversion, magField
) {
  postProgress(0.67, `Preparing ${dipoleMethod.toUpperCase()} dipole inversion...`);
  postLog(`Running ${dipoleMethod.toUpperCase()} dipole inversion...`);
  const progressCb = (current, total) => {
    postProgress(0.67 + (current / total) * 0.25, `${dipoleMethod.toUpperCase()}: ${current}/${total}`);
  };
  return runDipoleInversionByMethod(
    localField, erodedMask, nx, ny, nz, vsx, vsy, vsz,
    dipoleMethod, pipelineSettings,
    magnitudeData, echoTimes, skipHzConversion, magField,
    progressCb
  );
}

// =========================================================================
// SWI-only pipeline
// =========================================================================
async function runT2starR2starPipeline(data) {
  const {
    magnitudeBuffers, maskThreshold, customMaskBuffer, preparedMagnitude, echoTimes
  } = data;

  try {
    const nEchoes = magnitudeBuffers.length;
    if (nEchoes < 3) {
      throw new Error(`T2*/R2* mapping requires 3+ echoes, got ${nEchoes}`);
    }

    // Step 1: Load magnitude data
    postProgress(0.05, 'Loading magnitude data...');
    postLog(`T2*/R2*: Loading ${nEchoes} echo magnitudes...`);

    const magnitude4d = [];
    let dims, voxelSize, affine;
    for (let i = 0; i < nEchoes; i++) {
      const result = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[i]));
      magnitude4d.push(new Float64Array(result.data));
      if (i === 0) {
        dims = Array.from(result.dims);
        voxelSize = Array.from(result.voxelSize);
        affine = Array.from(result.affine);
      }
    }

    const [nx, ny, nz] = dims;
    const voxelCount = nx * ny * nz;
    postLog(`Data: ${nx}x${ny}x${nz}, ${nEchoes} echoes`);

    // Step 2: Create mask
    postProgress(0.15, 'Creating mask...');
    const thresholdFraction = (maskThreshold || 15) / 100;
    const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
    const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

    let mask;
    if (hasCustomMask) {
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else {
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : magnitude4d[0];
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    // Step 3: Compute R2* via ARLO
    postProgress(0.30, 'Computing R2* map...');
    postLog(`Computing R2* map (ARLO, ${nEchoes} echoes)...`);

    const interleaved = new Float64Array(voxelCount * nEchoes);
    for (let e = 0; e < nEchoes; e++) {
      for (let v = 0; v < voxelCount; v++) {
        interleaved[v * nEchoes + e] = magnitude4d[e][v];
      }
    }

    const echoTimesSec = new Float64Array(echoTimes.map(t => t / 1000));

    const r2starMap = new Float64Array(wasmModule.r2star_arlo_wasm(
      interleaved, mask, echoTimesSec, nx, ny, nz
    ));

    const r2range = computeRobustRange(r2starMap, mask, 2, 98);
    sendStageData('r2star', r2starMap, dims, voxelSize, affine, 'R2* Map (1/s)', true, r2range ? [0, r2range[1]] : null);
    postLog('R2* map computed');

    // Step 4: Compute T2* = 1/R2*
    postProgress(0.80, 'Computing T2* map...');
    const t2starMap = new Float64Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      t2starMap[i] = (mask[i] && r2starMap[i] > 0) ? (1.0 / r2starMap[i]) : 0;
    }
    const t2range = computeRobustRange(t2starMap, mask, 2, 90);
    sendStageData('t2star', t2starMap, dims, voxelSize, affine, 'T2* Map (s)', true, t2range ? [0, t2range[1]] : null);
    postLog('T2* map computed');

    postProgress(1.0, 'T2*/R2* complete!');
    postLog('T2*/R2* mapping completed successfully!');
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

async function runSWIPipeline(data) {
  const {
    magnitudeBuffers, phaseBuffers,
    maskThreshold, customMaskBuffer, preparedMagnitude, pipelineSettings
  } = data;

  const thresholdFraction = (maskThreshold || 15) / 100;
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;
  const hasPreparedMagnitude = preparedMagnitude !== null && preparedMagnitude !== undefined;

  try {
    // Step 1: Load first echo NIfTI data
    postProgress(0.05, 'Loading NIfTI data...');
    postLog("SWI: Loading data...");

    const magResult = wasmModule.load_nifti_wasm(new Uint8Array(magnitudeBuffers[0]));
    const magnitude = new Float64Array(magResult.data);
    const dims = Array.from(magResult.dims);
    const voxelSize = Array.from(magResult.voxelSize);
    const affine = Array.from(magResult.affine);

    const phaseResult = wasmModule.load_nifti_wasm(new Uint8Array(phaseBuffers[0]));
    let phase = scalePhase(new Float64Array(phaseResult.data));

    const [nx, ny, nz] = dims;
    const [vsx, vsy, vsz] = voxelSize;
    const voxelCount = nx * ny * nz;

    postLog(`Data: ${nx}x${ny}x${nz}, voxel: ${vsx.toFixed(2)}x${vsy.toFixed(2)}x${vsz.toFixed(2)}mm`);

    // Step 2: Create or load mask
    postProgress(0.15, 'Creating mask...');
    let mask;

    if (hasCustomMask) {
      const maskResult = wasmModule.load_nifti_wasm(new Uint8Array(customMaskBuffer));
      mask = new Uint8Array(voxelCount);
      for (let i = 0; i < voxelCount; i++) {
        mask[i] = maskResult.data[i] > 0.5 ? 1 : 0;
      }
    } else {
      const maskMagnitude = hasPreparedMagnitude
        ? new Float64Array(preparedMagnitude)
        : magnitude;
      mask = createThresholdMask(maskMagnitude, thresholdFraction);
    }

    const maskCount = mask.reduce((a, b) => a + b, 0);
    postLog(`Mask: ${maskCount}/${voxelCount} voxels (${(100 * maskCount / voxelCount).toFixed(1)}%)`);

    // Step 3: Laplacian phase unwrapping
    postProgress(0.25, 'Unwrapping phase...');
    postLog("Laplacian phase unwrapping...");
    const unwrappedPhase = new Float64Array(wasmModule.laplacian_unwrap_wasm(
      phase, mask, nx, ny, nz, vsx, vsy, vsz
    ));

    // Step 4: SWI computation
    postProgress(0.50, 'Computing SWI...');
    computeSWI(pipelineSettings, unwrappedPhase, magnitude, mask, dims, voxelSize, affine);

    postProgress(1.0, 'SWI complete!');
    postLog("SWI pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

// Handle messages from main thread
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init':
        await initializeWasm();
        self.postMessage({ type: 'initialized' });
        break;

      case 'run':
        await runPipeline(data);
        break;

      case 'runBET':
        await runBET(data);
        break;

      case 'runSWI':
        await runSWIPipeline(data);
        break;

      case 'runT2starR2star':
        await runT2starR2starPipeline(data);
        break;

      case 'biasCorrection':
        await runBiasCorrection(data);
        break;

      case 'voxelQuality':
        await runVoxelQuality(data);
        break;

      case 'getDefaultConfig':
        self.postMessage({ type: 'defaultConfig', result: wasmModule.get_default_config_json_wasm() });
        break;

      case 'generateCommand':
        self.postMessage({ type: 'commandResult', result: wasmModule.generate_command_wasm(data.configJson, data.maskSection || '') });
        break;

      case 'generateMethods':
        self.postMessage({ type: 'methodsResult', result: wasmModule.generate_methods_wasm(data.configJson, 'QSMbly', data.maskSection || '') });
        break;

      case 'generateConfigToml':
        // Download path: pruned to the selected algorithm (still loads in qsmxt.rs).
        self.postMessage({ type: 'configTomlResult', result: wasmModule.config_json_to_toml_selected_wasm(data.configJson, data.maskSection || '') });
        break;

      default:
        postError(`Unknown message type: ${type}`);
    }
  } catch (error) {
    postError(error.message);
    console.error(error);
  }
};
