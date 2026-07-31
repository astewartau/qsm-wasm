/**
 * QSM-WASM Configuration Module
 *
 * Centralized configuration for all magic numbers, default values,
 * and constants used across the application.
 *
 * This module works as both an ES module (for modern scripts) and
 * can be loaded via importScripts in web workers.
 */

// Detect environment and set up exports appropriately
const isWorker = typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;
const isModule = typeof exports !== 'undefined' || (typeof window !== 'undefined' && window.QSMConfig === undefined);

// Application version — 0.0.0 in git; the release CI stamps the real version (the release
// tag) into the build at deploy time. Single source of truth is the git tag.
export const VERSION = '0.0.0';

// QSM.rs core library version (the pinned qsm-core dependency tag in rust-wasm/Cargo.toml)
export const QSM_RS_VERSION = '0.23.0';

// Physics constants
export const PHYSICS = {
  GYROMAGNETIC_RATIO: 42.576e6  // Hz/T (gamma for protons)
};

// NiiVue viewer configuration
export const VIEWER_CONFIG = {
  loadingText: "",
  dragToMeasure: false,
  isColorbar: false,
  textHeight: 0.03,
  show3Dcrosshair: false,
  crosshairColor: [0.23, 0.51, 0.96, 1.0],  // #3b82f6 - matches site primary color
  crosshairWidth: 0.75
};

// Input mode configuration
export const INPUT_MODES = {
  RAW: 'raw',              // Raw magnitude + phase images (default)
  TOTAL_FIELD: 'totalField', // Pre-computed total field map (B0)
  LOCAL_FIELD: 'localField'  // Pre-computed local field map
};

export const FIELD_MAP_UNITS = {
  HZ: 'hz',       // Hertz
  RAD_S: 'rad_s',  // Radians per second
  PPM: 'ppm'       // Parts per million (already normalized)
};

export const INPUT_DEFAULTS = {
  inputMode: 'dicom',
  fieldMapUnits: 'hz'
};

// Algorithm defaults — sourced from QSM.rs via auto-generated qsm-defaults.js
// These re-exports adapt the QSM.rs field names to the JS conventions used throughout qsmbly.
import {
  RTS_DEFAULTS as _RTS,
  TV_DEFAULTS as _TV,
  TKD_DEFAULTS as _TKD,
  TGV_DEFAULTS as _TGV,
  BET_DEFAULTS as _BET,
  VSHARP_DEFAULTS as _VSHARP,
  PDF_DEFAULTS as _PDF,
  LBV_DEFAULTS as _LBV,
  ISMV_DEFAULTS as _ISMV,
  SWI_DEFAULTS as _SWI,
  SHARP_DEFAULTS as _SHARP,
  RESHARP_DEFAULTS as _RESHARP,
  HARPERELLA_DEFAULTS as _HARPERELLA,
  TIKHONOV_DEFAULTS as _TIKHONOV,
  NLTV_DEFAULTS as _NLTV,
  NDI_DEFAULTS as _NDI,
  FANSI_DEFAULTS as _FANSI,
  L1QSM_DEFAULTS as _L1QSM,
  WHQSM_DEFAULTS as _WHQSM,
  HDQSM_DEFAULTS as _HDQSM,
  MEDI_DEFAULTS as _MEDI,
  TFI_DEFAULTS as _TFI,
  QSMART_DEFAULTS as _QSMART,
  ROMEO_DEFAULTS as _ROMEO,
  MCPC3DS_DEFAULTS as _MCPC3DS,
  LINEAR_FIT_DEFAULTS as _LINEAR_FIT,
  HOMOGENEITY_DEFAULTS as _HOMOGENEITY,
} from './qsm-defaults.js';

// Re-export with JS-convention field names (camelCase, matching existing usage)
export const RTS_DEFAULTS = {
  delta: _RTS.delta,
  mu: _RTS.mu,
  rho: _RTS.rho,
  tol: _RTS.tol,
  max_iter: _RTS.max_iter,
  lsmr_iter: _RTS.lsmr_iter,
};

export const TV_DEFAULTS = {
  lambda: _TV.lambda,
  rho: _TV.rho,
  tol: _TV.tol,
  max_iter: _TV.max_iter,
};

export const TKD_DEFAULTS = {
  threshold: _TKD.threshold,
};

export const VSHARP_DEFAULTS = {
  threshold: _VSHARP.threshold,
  max_radius_factor: _VSHARP.max_radius_factor,
  min_radius_factor: _VSHARP.min_radius_factor,
};

export const PDF_DEFAULTS = {
  tol: _PDF.tol,
};

export const LBV_DEFAULTS = {
  tol: _LBV.tol,
};

export const ISMV_DEFAULTS = {
  tol: _ISMV.tol,
  maxit: _ISMV.max_iter,
  radius_factor: _ISMV.radius_factor,
};

// Adapted re-exports (field name mapping from snake_case to camelCase)
export const BET_DEFAULTS = {
  fractionalIntensity: _BET.fractional_intensity,
  smoothness: _BET.smoothness,
  gradientThreshold: _BET.gradient_threshold,
  iterations: _BET.iterations,
  subdivisions: _BET.subdivisions,
  erosions: 2  // mask erosion count (qsmbly-specific, not a BET param)
};

export const TGV_DEFAULTS = {
  regularization: 2,  // UI preset level (qsmbly-specific)
  alpha0: _TGV.alpha0,
  alpha1: _TGV.alpha1,
  iterations: _TGV.iterations,
  erosions: _TGV.erosions,
  step_size: _TGV.step_size,
  tol: _TGV.tol,
};

export const SWI_DEFAULTS = {
  hp_sigma: _SWI.hp_sigma,
  scaling: _SWI.scaling,
  strength: _SWI.strength,
  mip_window: _SWI.mip_window,
};

// Mask configuration defaults
export const MASK_CONFIG = {
  defaultThreshold: 15,         // Percentage of max magnitude
  drawingOpacity: 0.5,
  defaultBrushSize: 2
};

// Mask preparation settings
export const MASK_PREP_DEFAULTS = {
  source: 'phase_quality',      // 'first_echo', 'combined', or 'phase_quality'
  biasCorrection: true
};

// Progress animation settings
export const PROGRESS_CONFIG = {
  animationSpeed: 0.5           // 50% per second - catches up quickly
};

// (TGV_DEFAULTS and SWI_DEFAULTS are now sourced from QSM.rs — see imports above)

// QSMART pipeline defaults (from QSM.rs QsmartParams)
export const QSMART_DEFAULTS = {
  sdf_sigma1_stage1: _QSMART.sdf_sigma1_stage1,
  sdf_sigma2_stage1: _QSMART.sdf_sigma2_stage1,
  sdf_sigma1_stage2: _QSMART.sdf_sigma1_stage2,
  sdf_sigma2_stage2: _QSMART.sdf_sigma2_stage2,
  sdf_spatial_radius: _QSMART.sdf_spatial_radius,
  sdf_lower_lim: _QSMART.sdf_lower_lim,
  sdf_curv_constant: _QSMART.sdf_curv_constant,
  vasc_sphere_radius: _QSMART.vasc_sphere_radius,
  frangi_scale_min: _QSMART.frangi_scale_min,
  frangi_scale_max: _QSMART.frangi_scale_max,
  frangi_scale_ratio: _QSMART.frangi_scale_ratio,
  frangi_c: _QSMART.frangi_c,
  ilsqr_tol: _QSMART.ilsqr_tol,
  ilsqr_max_iter: _QSMART.ilsqr_max_iter,
  inversion_algorithm: _QSMART.inversion,
};

// ROMEO unwrapping defaults
export const ROMEO_DEFAULTS = {
  weighting: 'phase_snr',
  phase_gradient_coherence: _ROMEO.phase_gradient_coherence,
  mag_coherence: _ROMEO.mag_coherence,
  mag_weight: _ROMEO.mag_weight,
};

// MCPC-3D-S phase offset correction defaults
export const MCPC3DS_DEFAULTS = {
  sigma: _MCPC3DS.sigma,
};

// Linear fit B0 calculation defaults
export const LINEAR_FIT_DEFAULTS = {
  estimate_offset: _LINEAR_FIT.estimate_offset,
};

// Homogeneity correction defaults
export const HOMOGENEITY_DEFAULTS = {
  sigmaMm: _HOMOGENEITY.sigma_mm,
  nbox: _HOMOGENEITY.nbox,
};

export const SHARP_DEFAULTS = {
  threshold: _SHARP.threshold,
  radius_factor: _SHARP.radius_factor,
};

export const RESHARP_DEFAULTS = {
  radius: _RESHARP.radius,
  tik_reg: _RESHARP.tik_reg,
  tol: _RESHARP.tol,
  max_iter: _RESHARP.max_iter,
};

export const HARPERELLA_DEFAULTS = {
  radius: _HARPERELLA.radius,
  max_iter: _HARPERELLA.max_iter,
  tol: _HARPERELLA.tol,
};

// TSVD (Truncated SVD) defaults (shares TKD threshold)
export const TSVD_DEFAULTS = {
  threshold: _TKD.threshold,
};

// (VSHARP_DEFAULTS, ISMV_DEFAULTS, PDF_DEFAULTS, LBV_DEFAULTS,
//  TKD_DEFAULTS, RTS_DEFAULTS, TV_DEFAULTS are now sourced from QSM.rs
//  — see imports above)

// Tikhonov regularization defaults
export const TIKHONOV_DEFAULTS = {
  lambda: _TIKHONOV.lambda,
  reg: _TIKHONOV.reg,
};

// (TV_DEFAULTS, RTS_DEFAULTS sourced from QSM.rs — see imports above)

export const NLTV_DEFAULTS = {
  lambda: _NLTV.lambda,
  mu: _NLTV.mu,
  max_iter: _NLTV.max_iter,
  tol: _NLTV.tol,
  newton_max_iter: _NLTV.newton_iter,
};

export const NDI_DEFAULTS = {
  tau: _NDI.tau,
  alpha: _NDI.alpha,
  max_iter: _NDI.max_iter,
};

export const FANSI_DEFAULTS = {
  alpha1: _FANSI.alpha1,
  mu1: _FANSI.mu1,
  mu2: _FANSI.mu2,
  alpha0: _FANSI.alpha0,
  mu0: _FANSI.mu0,
  max_iter: _FANSI.max_iter,
  tol_update: _FANSI.tol_update,
};

export const L1QSM_DEFAULTS = {
  alpha1: _L1QSM.alpha1,
  mu1: _L1QSM.mu1,
  mu2: _L1QSM.mu2,
  mu3: _L1QSM.mu3,
  lambda: _L1QSM.lambda,
  max_iter: _L1QSM.max_iter,
  tol_update: _L1QSM.tol_update,
};

export const WHQSM_DEFAULTS = {
  alpha1: _WHQSM.alpha1,
  mu1: _WHQSM.mu1,
  mu2: _WHQSM.mu2,
  beta: _WHQSM.beta,
  muh: _WHQSM.muh,
  max_iter: _WHQSM.max_iter,
  tol_update: _WHQSM.tol_update,
};

export const HDQSM_DEFAULTS = {
  alpha_l2: _HDQSM.alpha_l2,
  mu1_l2: _HDQSM.mu1_l2,
  mu2: _HDQSM.mu2,
  max_iter_l1: _HDQSM.max_iter_l1,
  max_iter_l2: _HDQSM.max_iter_l2,
  tol_update: _HDQSM.tol_update,
};

export const MEDI_DEFAULTS = {
  lambda: _MEDI.lambda,
  percentage: _MEDI.percentage,
  max_iter: _MEDI.max_iter,
  cg_max_iter: _MEDI.cg_max_iter,
  cg_tol: _MEDI.cg_tol,
  tol: _MEDI.tol,
  smv: _MEDI.smv,
  smv_radius: _MEDI.smv_radius,
  merit: _MEDI.merit,
  data_weighting: _MEDI.data_weighting,
};

// TFI (preconditioned total field inversion) — single-step combined method (like TGV/QSMART).
export const TFI_DEFAULTS = {
  lambda: _TFI.lambda,
  precond: _TFI.precond,
  merit: _TFI.merit,
  data_weighting: _TFI.data_weighting,
  percentage: _TFI.percentage,
  cg_tol: _TFI.cg_tol,
  cg_max_iter: _TFI.cg_max_iter,
  max_iter: _TFI.max_iter,
  tol: _TFI.tol,
};

// Example data (downloaded from OSF during CI, served same-origin)
export const EXAMPLE_DATA = {
  baseUrl: './data/example',
  files: [
    'sub-1_echo-1_part-mag_MEGRE.nii.gz',
    'sub-1_echo-1_part-mag_MEGRE.json',
    'sub-1_echo-1_part-phase_MEGRE.nii.gz',
    'sub-1_echo-1_part-phase_MEGRE.json',
    'sub-1_echo-2_part-mag_MEGRE.nii.gz',
    'sub-1_echo-2_part-mag_MEGRE.json',
    'sub-1_echo-2_part-phase_MEGRE.nii.gz',
    'sub-1_echo-2_part-phase_MEGRE.json',
    'sub-1_echo-3_part-mag_MEGRE.nii.gz',
    'sub-1_echo-3_part-mag_MEGRE.json',
    'sub-1_echo-3_part-phase_MEGRE.nii.gz',
    'sub-1_echo-3_part-phase_MEGRE.json',
    'sub-1_echo-4_part-mag_MEGRE.nii.gz',
    'sub-1_echo-4_part-mag_MEGRE.json',
    'sub-1_echo-4_part-phase_MEGRE.nii.gz',
    'sub-1_echo-4_part-phase_MEGRE.json',
  ]
};

// Stage display names for UI
export const STAGE_DISPLAY_NAMES = {
  'magnitude': 'Magnitude',
  'phase': 'Phase',
  'mask': 'Mask',
  'B0': 'B0 Field',
  'bgRemoved': 'Local Field',
  'final': 'QSM',
  'tfs': 'TFS',
  'lfsStage1': 'LFS 1',
  'lfsStage2': 'LFS 2',
  'chiStage1': 'χ1',
  'chiStage2': 'χ2',
  'vasculature': 'Vessels',
  'vascDetect': 'Vessels',
  'frangi': 'Frangi',
  'vascMask': 'Vasc Mask',
  'bottomHat': 'Bottom Hat',
  'unwrapped': 'Unwrapped',
  'swi': 'SWI',
  'mip': 'mIP',
  'r2star': 'R2*',
  't2star': 'T2*'
};

// Pipeline method options
export const PIPELINE_METHODS = {
  combined: ['none', 'tgv', 'qsmart', 'tfi'],
  unwrap: ['romeo', 'laplacian'],
  phaseOffset: ['mcpc3ds', 'none'],
  fieldCalculation: ['weighted_avg', 'linear_fit'],
  b0_weight_type: ['phase_snr', 'phase_var', 'average', 'tes', 'mag'],
  bf_algorithm: ['vsharp', 'sharp', 'resharp', 'ismv', 'pdf', 'lbv', 'harperella', 'iharperella'],
  dipole_inversion: ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi', 'ndi', 'fansi', 'fansitgv', 'l1qsm', 'whqsm', 'hdqsm'],
  // Inner dipole inversion for QSMART's two stages (includes ilsqr, excludes tgv/qsmart)
  qsmart_inversion: ['tkd', 'tsvd', 'tikhonov', 'tv', 'rts', 'nltv', 'medi', 'ilsqr']
};

// Default pipeline settings (assembled from individual defaults)
export const PIPELINE_DEFAULTS = {
  combined_method: 'none',
  swi: { ...SWI_DEFAULTS },
  tgv: { ...TGV_DEFAULTS },
  qsmart: { ...QSMART_DEFAULTS },
  tfi: { ...TFI_DEFAULTS },
  unwrapping_algorithm: 'romeo',
  phase_offset_method: 'mcpc3ds',
  b0_estimation: 'weighted_avg',
  mcpc3ds: { ...MCPC3DS_DEFAULTS },
  b0_weight_type: 'phase_snr',
  linearFit: { ...LINEAR_FIT_DEFAULTS },
  romeo: { ...ROMEO_DEFAULTS },
  bf_algorithm: 'vsharp',
  vsharp: { ...VSHARP_DEFAULTS, max_radius: null, min_radius: null },
  sharp: { radius: 6, ...SHARP_DEFAULTS },
  resharp: { ...RESHARP_DEFAULTS },
  ismv: { ...ISMV_DEFAULTS, radius: null },
  pdf: { ...PDF_DEFAULTS, maxit: null },
  lbv: { ...LBV_DEFAULTS },
  harperella: { ...HARPERELLA_DEFAULTS },
  iharperella: { ...HARPERELLA_DEFAULTS },
  dipole_inversion: 'rts',
  tkd: { ...TKD_DEFAULTS },
  tsvd: { ...TSVD_DEFAULTS },
  tikhonov: { ...TIKHONOV_DEFAULTS },
  tv: { ...TV_DEFAULTS },
  rts: { ...RTS_DEFAULTS },
  nltv: { ...NLTV_DEFAULTS },
  ndi: { ...NDI_DEFAULTS },
  fansi: { ...FANSI_DEFAULTS },
  fansitgv: { ...FANSI_DEFAULTS },
  l1qsm: { ...L1QSM_DEFAULTS },
  whqsm: { ...WHQSM_DEFAULTS },
  hdqsm: { ...HDQSM_DEFAULTS },
  medi: { ...MEDI_DEFAULTS }
};

/**
 * Calculate dynamic defaults based on voxel size
 * Matches QSM.jl behavior for radius calculations
 *
 * @param {number[]} voxelSize - [dx, dy, dz] in mm
 * @param {number[]} maskDims - [nx, ny, nz] dimensions (optional, for PDF maxit)
 * @returns {Object} Calculated defaults
 */
export function getVoxelBasedDefaults(voxelSize = [1, 1, 1], maskDims = null) {
  const minVsz = Math.min(...voxelSize);
  const maxVsz = Math.max(...voxelSize);

  // Calculate mask size for PDF maxit (if available)
  const maskSize = maskDims ? maskDims[0] * maskDims[1] * maskDims[2] : 100000;

  // Max dimension for LBV adaptive maxit
  const maxDim = maskDims ? Math.max(maskDims[0], maskDims[1], maskDims[2]) : 256;

  return {
    // V-SHARP: max_radius = 18 * min(vsz), min_radius = 2 * max(vsz)
    vsharpMaxRadius: Math.round(18 * minVsz),
    vsharpMinRadius: Math.round(Math.max(1, 2 * minVsz)),
    // SHARP: radius = 18 * min(vsz) - matches QSM.jl sharp.jl line 22
    sharpRadius: Math.round(18 * minVsz),
    // iSMV: radius = 2 * max(vsz) - matches QSM.jl ismv.jl line 20
    ismv_radius: Math.round(Math.max(2, 2 * maxVsz)),
    // PDF: maxit = ceil(sqrt(numel(mask))) - matches QSM.jl pdf.jl
    pdfMaxit: Math.ceil(Math.sqrt(maskSize)),
    // LBV: maxit = max(dims) - matches QSM.jl lbv.jl
    lbvMaxit: maxDim
  };
}

/**
 * Phase scaling constants
 */
export const PHASE_SCALING = {
  PI_THRESHOLD_MULTIPLIER: 1.1,   // Range > 2π * 1.1 triggers scaling
  MAX_PI_MULTIPLIER: 1.5          // Values > π * 1.5 trigger scaling
};

/**
 * Box filter default radius for reliability map computation
 */
export const BOX_FILTER_DEFAULTS = {
  reliabilityRadius: 1
};

// Make config available globally for non-module scripts and workers
const QSMConfig = {
  VERSION,
  QSM_RS_VERSION,
  PHYSICS,
  INPUT_MODES,
  FIELD_MAP_UNITS,
  INPUT_DEFAULTS,
  VIEWER_CONFIG,
  MASK_CONFIG,
  BET_DEFAULTS,
  MASK_PREP_DEFAULTS,
  PROGRESS_CONFIG,
  SWI_DEFAULTS,
  TGV_DEFAULTS,
  QSMART_DEFAULTS,
  TFI_DEFAULTS,
  ROMEO_DEFAULTS,
  MCPC3DS_DEFAULTS,
  LINEAR_FIT_DEFAULTS,
  VSHARP_DEFAULTS,
  SHARP_DEFAULTS,
  RESHARP_DEFAULTS,
  HARPERELLA_DEFAULTS,
  ISMV_DEFAULTS,
  PDF_DEFAULTS,
  LBV_DEFAULTS,
  TKD_DEFAULTS,
  TSVD_DEFAULTS,
  TIKHONOV_DEFAULTS,
  TV_DEFAULTS,
  RTS_DEFAULTS,
  NLTV_DEFAULTS,
  NDI_DEFAULTS,
  FANSI_DEFAULTS,
  L1QSM_DEFAULTS,
  WHQSM_DEFAULTS,
  HDQSM_DEFAULTS,
  MEDI_DEFAULTS,
  EXAMPLE_DATA,
  STAGE_DISPLAY_NAMES,
  PIPELINE_METHODS,
  PIPELINE_DEFAULTS,
  getVoxelBasedDefaults,
  PHASE_SCALING,
  BOX_FILTER_DEFAULTS
};

// Export for different environments
if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined') {
  // Web Worker context
  self.QSMConfig = QSMConfig;
} else if (typeof window !== 'undefined') {
  // Browser context
  window.QSMConfig = QSMConfig;
}
