//! QSM-WASM: WebAssembly bindings for QSM processing
//!
//! This crate provides #[wasm_bindgen] wrappers around qsm-core algorithms
//! for browser-based medical image processing.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Gyromagnetic ratio of hydrogen protons (Hz/T)
const GYROMAGNETIC_RATIO: f64 = 42.576e6;

/// Returns scale factor to convert Hz → ppm given field strength in Tesla.
/// Returns 1.0 if field_strength <= 0 (no conversion).
fn hz_to_ppm_scale(field_strength: f64) -> f64 {
    if field_strength > 0.0 {
        1e6 / (GYROMAGNETIC_RATIO * field_strength)
    } else {
        1.0
    }
}

// ============================================================================
// WASM Exports: Phase Unwrapping
// ============================================================================

/// WASM-accessible region growing phase unwrapping
///
/// # Arguments
/// * `phase` - Float64Array of phase values (nx * ny * nz), modified in-place
/// * `weights` - Uint8Array of weights (3 * nx * ny * nz), layout [dim][x][y][z]
/// * `mask` - Uint8Array mask (nx * ny * nz), 1 = process, 0 = skip (modified: 2 = visited)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `seed_i`, `seed_j`, `seed_k` - Seed point coordinates
///
/// # Returns
/// Number of voxels processed
#[wasm_bindgen]
pub fn grow_region_unwrap_wasm(
    phase: &mut [f64],
    weights: &[u8],
    mask: &mut [u8],
    nx: usize,
    ny: usize,
    nz: usize,
    seed_i: usize,
    seed_j: usize,
    seed_k: usize,
) -> usize {
    console_log!("WASM grow_region_unwrap: {}x{}x{}, seed=({},{},{})",
                 nx, ny, nz, seed_i, seed_j, seed_k);

    let processed = qsm_core::region_grow::grow_region_unwrap(
        phase, weights, mask, nx, ny, nz, seed_i, seed_j, seed_k
    );

    console_log!("WASM processed {} voxels", processed);
    processed
}

/// Laplacian phase unwrapping
///
/// Uses FFT-based Poisson solver - fast but may have issues at mask boundaries.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
///
/// # Returns
/// Unwrapped phase
#[wasm_bindgen]
pub fn laplacian_unwrap_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    console_log!("WASM laplacian_unwrap: {}x{}x{}", nx, ny, nz);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let unwrapped = qsm_core::unwrap::laplacian_unwrap(phase, mask, &grid);

    console_log!("WASM laplacian_unwrap complete");
    unwrapped
}

/// Calculate ROMEO edge weights for phase unwrapping
///
/// # Arguments
/// * `phase` - Phase data (nx * ny * nz)
/// * `mag` - Magnitude data (nx * ny * nz), can be empty
/// * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Weights array (3 * nx * ny * nz) for x, y, z directions
#[wasm_bindgen]
pub fn calculate_weights_romeo_wasm(
    phase: &[f64],
    mag: &[f64],
    phase2: &[f64],
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<u8> {
    console_log!("WASM calculate_weights_romeo: {}x{}x{}", nx, ny, nz);

    let phase2_opt = if phase2.is_empty() { None } else { Some(phase2) };

    let weights = qsm_core::unwrap::romeo::calculate_weights_romeo(
        phase, mag, phase2_opt, te1, te2, mask, nx, ny, nz
    );

    console_log!("WASM weights calculation complete");
    weights
}

/// Calculate ROMEO edge weights with configurable weight components
///
/// # Arguments
/// * `phase` - Phase data (nx * ny * nz)
/// * `mag` - Magnitude data (nx * ny * nz), can be empty
/// * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `use_phase_gradient_coherence` - Include phase gradient coherence (multi-echo temporal)
/// * `use_mag_coherence` - Include magnitude coherence (min/max similarity)
/// * `use_mag_weight` - Include magnitude weight (penalize low signal)
///
/// # Returns
/// Weights array (3 * nx * ny * nz) for x, y, z directions
#[wasm_bindgen]
pub fn calculate_weights_romeo_configurable_wasm(
    phase: &[f64],
    mag: &[f64],
    phase2: &[f64],
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    use_phase_gradient_coherence: bool,
    use_mag_coherence: bool,
    use_mag_weight: bool,
) -> Vec<u8> {
    console_log!("WASM calculate_weights_romeo_configurable: {}x{}x{}, pgc={}, mc={}, mw={}",
                 nx, ny, nz, use_phase_gradient_coherence, use_mag_coherence, use_mag_weight);

    let phase2_opt = if phase2.is_empty() { None } else { Some(phase2) };

    let flags = [
        true,                          // phase coherence (always on)
        use_phase_gradient_coherence,  // phase gradient coherence
        false,                         // phase linearity
        use_mag_coherence,             // magnitude coherence
        use_mag_weight,                // magnitude weight
        false,                         // magnitude weight 2
    ];
    let weights = qsm_core::unwrap::romeo::calculate_weights_romeo_with_flags(
        phase, mag, phase2_opt, te1, te2, mask, nx, ny, nz,
        flags
    );

    console_log!("WASM weights calculation complete");
    weights
}

/// Calculate ROMEO voxel quality map for phase-based masking
///
/// Computes per-voxel quality by averaging ROMEO edge weights across all
/// 6 neighboring directions. Values range from 0 to 100.
///
/// # Arguments
/// * `phase` - Phase data (nx * ny * nz)
/// * `mag` - Magnitude data (nx * ny * nz), can be empty
/// * `phase2` - Second echo phase for gradient coherence (nx * ny * nz), can be empty
/// * `te1`, `te2` - Echo times for gradient coherence scaling
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Quality map (nx * ny * nz) with values in range [0, 100]
#[wasm_bindgen]
pub fn voxel_quality_romeo_wasm(
    phase: &[f64],
    mag: &[f64],
    phase2: &[f64],
    te1: f64,
    te2: f64,
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    console_log!("WASM voxel_quality_romeo: {}x{}x{}", nx, ny, nz);

    let phase2_opt = if phase2.is_empty() { None } else { Some(phase2) };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let quality = qsm_core::unwrap::romeo::voxel_quality_romeo(
        phase, mag, phase2_opt, te1, te2, mask, &grid
    );

    console_log!("WASM voxel quality map complete");
    quality
}

// ============================================================================
// WASM Exports: Dipole Inversion
// ============================================================================

/// TKD (Truncated K-space Division) dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside, 0 = outside
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `threshold` - TKD threshold (typically 0.1-0.2)
///
/// # Returns
/// Susceptibility map as Float64Array
#[wasm_bindgen]
pub fn tkd_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    threshold: f64,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM TKD: {}x{}x{}, thr={:.3}, scale={:.4e}",
                 nx, ny, nz, threshold, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::tkd(
        &field_norm, mask, &grid,
        (bx, by, bz), &qsm_core::inversion::TkdParams { threshold },
    );

    console_log!("WASM TKD complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// TSVD (Truncated SVD) dipole inversion
///
/// Similar to TKD but zeros values below threshold instead of truncating.
#[wasm_bindgen]
pub fn tsvd_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    threshold: f64,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM TSVD: {}x{}x{}, scale={:.4e}", nx, ny, nz, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::tsvd(
        &field_norm, mask, &grid,
        (bx, by, bz), &qsm_core::inversion::TkdParams { threshold },
    );

    console_log!("WASM TSVD complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// Tikhonov regularized dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter
/// * `reg_type` - Regularization type: 0=identity, 1=gradient, 2=laplacian
#[wasm_bindgen]
pub fn tikhonov_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    reg_type: u8,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM Tikhonov: {}x{}x{}, lambda={:.4}, reg_type={}, scale={:.4e}",
                 nx, ny, nz, lambda, reg_type, scale);

    let reg = match reg_type {
        0 => qsm_core::inversion::tikhonov::Regularization::Identity,
        1 => qsm_core::inversion::tikhonov::Regularization::Gradient,
        _ => qsm_core::inversion::tikhonov::Regularization::Laplacian,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::TikhonovParams { lambda, reg };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::tikhonov(
        &field_norm, mask, &grid,
        (bx, by, bz), &params
    );

    console_log!("WASM Tikhonov complete");
    chi.iter().map(|&v| v / scale).collect()
}

// ============================================================================
// WASM Exports: Background Field Removal
// ============================================================================

/// Build `VsharpParams` from an explicit list of mm radii.
///
/// qsm-core takes absolute mm radii directly: `max_radius` is the largest
/// radius and `min_radius` is the step between successive radii (V-SHARP radii
/// are uniformly spaced), which is also the smallest kernel.
fn vsharp_params_from_radii(
    radii: &[f64], threshold: f64, _vsx: f64, _vsy: f64, _vsz: f64,
) -> qsm_core::bgremove::VsharpParams {
    let d = qsm_core::bgremove::VsharpParams::default();
    if radii.is_empty() {
        return qsm_core::bgremove::VsharpParams { threshold, ..d };
    }
    let max_r = radii.iter().cloned().fold(f64::MIN, f64::max);
    // Recover the step from the two largest radii; a single-radius list has no
    // step, so fall back to the radius itself (one-shot SMV).
    let step = if radii.len() >= 2 {
        let mut desc = radii.to_vec();
        desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
        (desc[0] - desc[1]).abs().max(f64::EPSILON)
    } else {
        max_r
    };
    qsm_core::bgremove::VsharpParams {
        threshold,
        max_radius: max_r,
        min_radius: step,
    }
}

/// SHARP background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `threshold` - High-pass filter threshold
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64 for simplicity)
#[wasm_bindgen]
pub fn sharp_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    threshold: f64,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM SHARP: {}x{}x{}, radius={:.1}, field_strength={:.1}T, scale={:.4e}",
                 nx, ny, nz, radius, field_strength, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let (local_field, eroded_mask) = qsm_core::bgremove::sharp(
        &field_norm, mask, &grid,
        &qsm_core::bgremove::SharpParams { threshold, radius },
    );

    // Convert back and combine into single output
    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM SHARP complete");
    result
}

/// Simple SMV background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn smv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
) -> Vec<f64> {
    console_log!("WASM SMV: {}x{}x{}, radius={:.1}", nx, ny, nz, radius);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let (local_field, eroded_mask) = qsm_core::bgremove::smv(
        field, mask, &grid, radius
    );

    // Combine into single output: local_field followed by mask as f64
    let mut result = local_field;
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM SMV complete");
    result
}

/// V-SHARP background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radii` - SMV kernel radii in mm (should be sorted large to small)
/// * `threshold` - High-pass filter threshold
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn vsharp_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM V-SHARP: {}x{}x{}, {} radii, scale={:.4e}", nx, ny, nz, radii.len(), scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let vsp = vsharp_params_from_radii(radii, threshold, vsx, vsy, vsz);
    let (local_field, eroded_mask) = qsm_core::bgremove::vsharp(
        &field_norm, mask, &grid, &vsp, |_, _| {}
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM V-SHARP complete");
    result
}

/// V-SHARP with progress callback
#[wasm_bindgen]
pub fn vsharp_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM V-SHARP with progress: {}x{}x{}, {} radii, scale={:.4e}", nx, ny, nz, radii.len(), scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let vsp = vsharp_params_from_radii(radii, threshold, vsx, vsy, vsz);
    let (local_field, eroded_mask) = qsm_core::bgremove::vsharp(
        &field_norm, mask, &grid, &vsp,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM V-SHARP complete");
    result
}

/// TV-ADMM regularized dipole inversion
///
/// Total Variation regularization using ADMM for edge-preserving QSM.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3 to 1e-4)
/// * `rho` - ADMM penalty parameter (typically 100*lambda)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
#[wasm_bindgen]
pub fn tv_admm_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM TV-ADMM: {}x{}x{}, lambda={:.4}, rho={:.4}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, lambda, rho, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::TvParams { lambda, rho, tol, max_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::tv_admm(
        &field_norm, mask, &grid,
        (bx, by, bz), &params, |_, _| {}
    );

    console_log!("WASM TV-ADMM complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// TV-ADMM with progress callback
#[wasm_bindgen]
pub fn tv_admm_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM TV-ADMM with progress: {}x{}x{}, lambda={:.4}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, lambda, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::TvParams { lambda, rho, tol, max_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::tv_admm(
        &field_norm, mask, &grid,
        (bx, by, bz), &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM TV-ADMM complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// RTS (Rapid Two-Step) dipole inversion
///
/// Two-step method: LSMR for well-conditioned k-space + TV for ill-conditioned.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `delta` - Threshold for ill-conditioned k-space (typically 0.15)
/// * `mu` - Regularization for well-conditioned (typically 1e5)
/// * `rho` - ADMM penalty parameter (typically 10)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations
/// * `lsmr_iter` - LSMR iterations for step 1
#[wasm_bindgen]
pub fn rts_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM RTS: {}x{}x{}, delta={:.2}, mu={:.0}, scale={:.4e}", nx, ny, nz, delta, mu, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::RtsParams { delta, mu, rho, tol, max_iter, lsmr_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::rts(
        &field_norm, mask, &grid,
        (bx, by, bz), &params, |_, _| {}
    );

    console_log!("WASM RTS complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// RTS with progress callback
#[wasm_bindgen]
pub fn rts_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM RTS with progress: {}x{}x{}, delta={:.2}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, delta, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::RtsParams { delta, mu, rho, tol, max_iter, lsmr_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::rts(
        &field_norm, mask, &grid,
        (bx, by, bz), &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM RTS complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// NLTV (Nonlinear Total Variation) dipole inversion
///
/// Iteratively reweighted TV for edge-preserving QSM.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3)
/// * `mu` - Reweighting parameter (typically 1.0)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations per reweighting step
/// * `newton_iter` - Number of reweighting steps
#[wasm_bindgen]
pub fn nltv_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM NLTV: {}x{}x{}, lambda={:.4}, mu={:.2}, max_iter={}, newton={}, scale={:.4e}",
                 nx, ny, nz, lambda, mu, max_iter, newton_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::NltvParams { lambda, mu, tol, max_iter, newton_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let chi = qsm_core::inversion::nltv(
        &field_norm, mask, &grid,
        (bx, by, bz), &params, |_, _| {}
    );

    console_log!("WASM NLTV complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// NLTV with progress callback
#[wasm_bindgen]
pub fn nltv_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM NLTV with progress: {}x{}x{}, lambda={:.4}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, lambda, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::NltvParams { lambda, mu, tol, max_iter, newton_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::nltv(
        &field_norm, mask, &grid,
        (bx, by, bz), &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM NLTV complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// MEDI L1 dipole inversion
///
/// Morphology-enabled dipole inversion with L1 TV regularization.
/// Features gradient weighting from magnitude, SNR-based data weighting,
/// optional SMV preprocessing, and optional merit-based outlier adjustment.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `n_std` - Noise standard deviation map (nx * ny * nz)
/// * `magnitude` - Magnitude image for edge weighting (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `lambda` - Regularization parameter (default 7.5e-5, matching MATLAB MEDI)
/// * `merit` - Enable merit-based outlier adjustment
/// * `smv` - Enable SMV preprocessing within MEDI
/// * `smv_radius` - SMV radius in mm (default 5.0)
/// * `data_weighting` - 0=uniform, 1=SNR weighting
/// * `percentage` - Fraction of voxels considered edges (default 0.3 = 30%)
/// * `cg_tol` - CG solver tolerance
/// * `cg_max_iter` - CG maximum iterations
/// * `max_iter` - Maximum Gauss-Newton iterations
/// * `tol` - Convergence tolerance
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_wasm(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    console_log!("WASM MEDI: {}x{}x{}, lambda={:.0}, max_iter={}, smv={}, merit={}",
                 nx, ny, nz, lambda, max_iter, smv, merit);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::MediParams {
        lambda, merit, smv, smv_radius, data_weighting, percentage,
        cg_tol, cg_max_iter, max_iter, tol,
    };
    let chi = qsm_core::inversion::medi(
        local_field, n_std, magnitude, mask, &grid,
        (bx, by, bz), &params, |_, _| {}
    );

    console_log!("WASM MEDI complete");
    chi
}

/// MEDI L1 with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn medi_l1_wasm_with_progress(
    local_field: &[f64],
    n_std: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    lambda: f64,
    merit: bool,
    smv: bool,
    smv_radius: f64,
    data_weighting: i32,
    percentage: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    max_iter: usize,
    tol: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM MEDI with progress: {}x{}x{}, lambda={:.0}, max_iter={}",
                 nx, ny, nz, lambda, max_iter);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::MediParams {
        lambda, merit, smv, smv_radius, data_weighting, percentage,
        cg_tol, cg_max_iter, max_iter, tol,
    };
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::medi(
        local_field, n_std, magnitude, mask, &grid,
        (bx, by, bz), &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM MEDI complete");
    chi
}

/// iLSQR dipole inversion with streaking artifact removal
///
/// A method for estimating and removing streaking artifacts in QSM.
/// Based on Li et al., NeuroImage 2015.
///
/// The algorithm consists of 4 steps:
/// 1. Initial LSQR solution with Laplacian-based weights
/// 2. FastQSM estimate using sign(D) approximation
/// 3. Streaking artifact estimation using LSMR
/// 4. Artifact subtraction
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `tol` - Stopping tolerance for LSMR solver (default 1e-2)
/// * `max_iter` - Maximum iterations for LSMR (default 50)
///
/// # Returns
/// Susceptibility map as Float64Array
#[wasm_bindgen]
pub fn ilsqr_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM iLSQR: {}x{}x{}, tol={:.4}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, tol, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::IlsqrParams { tol, max_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let (chi, _, _, _) = qsm_core::inversion::ilsqr(
        &field_norm, mask, &grid,
        (bx, by, bz), &params, |_, _| {},
    );

    console_log!("WASM iLSQR complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// iLSQR with progress callback
#[wasm_bindgen]
pub fn ilsqr_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM iLSQR with progress: {}x{}x{}, tol={:.4}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, tol, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::IlsqrParams { tol, max_iter };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::ilsqr(
        &field_norm, mask, &grid,
        (bx, by, bz), &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    ).0;

    console_log!("WASM iLSQR complete");
    chi.iter().map(|&v| v / scale).collect()
}

/// iLSQR with full output (susceptibility, artifacts, fastqsm, initial lsqr)
///
/// Returns all intermediate results for analysis/debugging.
///
/// # Returns
/// Flattened array: [chi, xsa, xfs, xlsqr] - 4 * (nx * ny * nz) elements
/// - chi: Final susceptibility map
/// - xsa: Estimated streaking artifacts
/// - xfs: FastQSM estimate
/// - xlsqr: Initial LSQR result
#[wasm_bindgen]
pub fn ilsqr_full_wasm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    console_log!("WASM iLSQR full: {}x{}x{}", nx, ny, nz);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::IlsqrParams { tol, max_iter };
    let (chi, xsa, xfs, xlsqr) = qsm_core::inversion::ilsqr(
        local_field, mask, &grid,
        (bx, by, bz), &params, |_, _| {}
    );

    // Concatenate all outputs
    let n_total = nx * ny * nz;
    let mut result = Vec::with_capacity(4 * n_total);
    result.extend(chi);
    result.extend(xsa);
    result.extend(xfs);
    result.extend(xlsqr);

    console_log!("WASM iLSQR full complete");
    result
}

// ============================================================================
// WASM Exports: TGV (Single-Step QSM from Wrapped Phase)
// ============================================================================

/// TGV-QSM (Total Generalized Variation) single-step reconstruction
///
/// Reconstructs susceptibility directly from wrapped phase data using TGV
/// regularization. This bypasses phase unwrapping and background field removal.
///
/// # Arguments
/// * `phase` - Wrapped phase data in radians (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `alpha0` - TGV second-order weight (symmetric gradient term)
/// * `alpha1` - TGV first-order weight (gradient term)
/// * `iterations` - Number of primal-dual iterations
/// * `erosions` - Number of mask erosions (default 3)
/// * `te` - Echo time in seconds
/// * `fieldstrength` - Magnetic field strength in Tesla
///
/// # Returns
/// Susceptibility map as Float64Array (ppm)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn tgv_qsm_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha0: f64,
    alpha1: f64,
    iterations: usize,
    erosions: usize,
    te: f64,
    fieldstrength: f64,
) -> Vec<f64> {
    console_log!("WASM TGV-QSM: {}x{}x{}, alpha=({:.4},{:.4}), iter={}, TE={}ms, B0={}T",
                 nx, ny, nz, alpha0, alpha1, iterations, te * 1000.0, fieldstrength);

    let params = qsm_core::inversion::tgv::TgvParams {
        alpha0: alpha0 as f32,
        alpha1: alpha1 as f32,
        iterations,
        erosions,
        step_size: 3.0,
        fieldstrength: fieldstrength as f32,
        te: te as f32,
        tol: 1e-5,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let chi = qsm_core::inversion::tgv_qsm(
        phase, mask, &grid,
        &params, (bx, by, bz),
        |_, _| {}
    );

    console_log!("WASM TGV-QSM complete");

    // Convert back to f64
    chi.iter().map(|&x| x as f64).collect()
}

/// TGV-QSM with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn tgv_qsm_wasm_with_progress(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha0: f64,
    alpha1: f64,
    iterations: usize,
    erosions: usize,
    te: f64,
    fieldstrength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM TGV-QSM with progress: {}x{}x{}, iter={}",
                 nx, ny, nz, iterations);

    let params = qsm_core::inversion::tgv::TgvParams {
        alpha0: alpha0 as f32,
        alpha1: alpha1 as f32,
        iterations,
        erosions,
        step_size: 3.0,
        fieldstrength: fieldstrength as f32,
        te: te as f32,
        tol: 1e-5,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let callback = progress_callback.clone();
    let chi = qsm_core::inversion::tgv_qsm(
        phase, mask, &grid,
        &params, (bx, by, bz),
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM TGV-QSM complete");
    chi.iter().map(|&x| x as f64).collect()
}

/// Get default TGV alpha values for a given regularization level (1-4)
/// Returns [alpha0, alpha1]
#[wasm_bindgen]
pub fn tgv_get_default_alpha(regularization: u8) -> Vec<f64> {
    let (alpha0, alpha1) = qsm_core::inversion::tgv::get_default_alpha(regularization);
    vec![alpha0 as f64, alpha1 as f64]
}

/// Get default TGV iteration count based on voxel size and step size.
/// Matches Julia reference: max(1000, 3200 / prod(res)^0.42) / step_size^0.6
#[wasm_bindgen]
pub fn tgv_get_default_iterations(vsx: f32, vsy: f32, vsz: f32, step_size: f32) -> usize {
    qsm_core::inversion::tgv::get_default_iterations((vsx, vsy, vsz), step_size)
}

// ============================================================================
// WASM Exports: Background Field Removal (continued)
// ============================================================================

/// PDF background field removal
///
/// Projection onto dipole fields for background removal.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `tol` - LSMR convergence tolerance
/// * `max_iter` - Maximum LSMR iterations
#[wasm_bindgen]
pub fn pdf_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM PDF: {}x{}x{}, scale={:.4e}", nx, ny, nz, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let local_field = qsm_core::bgremove::pdf(
        &field_norm, mask, &grid,
        (bx, by, bz),
        &qsm_core::bgremove::PdfParams { tol, max_iter: Some(max_iter) },
        |_, _| {}
    );

    console_log!("WASM PDF complete");
    local_field.iter().map(|&v| v / scale).collect()
}

/// PDF with progress callback
#[wasm_bindgen]
pub fn pdf_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM PDF with progress: {}x{}x{}, max_iter={}, scale={:.4e}", nx, ny, nz, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let local_field = qsm_core::bgremove::pdf(
        &field_norm, mask, &grid,
        (bx, by, bz),
        &qsm_core::bgremove::PdfParams { tol, max_iter: Some(max_iter) },
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM PDF complete");
    local_field.iter().map(|&v| v / scale).collect()
}

/// iSMV background field removal
///
/// Iterative SMV that preserves mask better than SHARP.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn ismv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM iSMV: {}x{}x{}, radius={:.1}, scale={:.4e}", nx, ny, nz, radius, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let ismv_params = qsm_core::bgremove::IsmvParams {
        tol, max_iter, radius,
    };
    let (local_field, eroded_mask) = qsm_core::bgremove::ismv(
        &field_norm, mask, &grid, &ismv_params, |_, _| {}
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM iSMV complete");
    result
}

/// iSMV with progress callback
#[wasm_bindgen]
pub fn ismv_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM iSMV with progress: {}x{}x{}, radius={:.1}, max_iter={}, scale={:.4e}",
                 nx, ny, nz, radius, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let ismv_params = qsm_core::bgremove::IsmvParams {
        tol, max_iter, radius,
    };
    let (local_field, eroded_mask) = qsm_core::bgremove::ismv(
        &field_norm, mask, &grid, &ismv_params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM iSMV complete");
    result
}

/// LBV (Laplacian Boundary Value) background field removal
///
/// Solves Laplace equation inside mask with Dirichlet boundary conditions.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Flattened array: first nx*ny*nz elements are local field,
/// next nx*ny*nz elements are eroded mask (as f64)
#[wasm_bindgen]
pub fn lbv_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM LBV: {}x{}x{}, tol={:.6}, max_iter={}, scale={:.4e}", nx, ny, nz, tol, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let (local_field, eroded_mask) = qsm_core::bgremove::lbv(
        &field_norm, mask, &grid,
        &qsm_core::bgremove::LbvParams { tol, max_iter: Some(max_iter) },
        |_, _| {}
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM LBV complete");
    result
}

/// LBV with progress callback
#[wasm_bindgen]
pub fn lbv_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM LBV with progress: {}x{}x{}, tol={:.6}, max_iter={}, scale={:.4e}", nx, ny, nz, tol, max_iter, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let (local_field, eroded_mask) = qsm_core::bgremove::lbv(
        &field_norm, mask, &grid,
        &qsm_core::bgremove::LbvParams { tol, max_iter: Some(max_iter) },
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM LBV complete");
    result
}

/// RESHARP background field removal
#[wasm_bindgen]
pub fn resharp_wasm(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tik_reg: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM RESHARP: {}x{}x{}, radius={:.1}, tik_reg={:.1e}, scale={:.4e}",
                 nx, ny, nz, radius, tik_reg, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::bgremove::ResharpParams { radius, tik_reg, tol, max_iter };
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let (local_field, eroded_mask) = qsm_core::bgremove::resharp(
        &field_norm, mask, &grid, &params, |_, _| {}
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM RESHARP complete");
    result
}

/// RESHARP with progress callback
#[wasm_bindgen]
pub fn resharp_wasm_with_progress(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    tik_reg: f64,
    tol: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM RESHARP with progress: {}x{}x{}, radius={:.1}, tik_reg={:.1e}, scale={:.4e}",
                 nx, ny, nz, radius, tik_reg, scale);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::bgremove::ResharpParams { radius, tik_reg, tol, max_iter };
    let field_norm: Vec<f64> = field.iter().map(|&v| v * scale).collect();
    let callback = progress_callback.clone();
    let (local_field, eroded_mask) = qsm_core::bgremove::resharp(
        &field_norm, mask, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result: Vec<f64> = local_field.iter().map(|&v| v / scale).collect();
    result.extend(eroded_mask.iter().map(|&m| m as f64));

    console_log!("WASM RESHARP complete");
    result
}

/// HARPERELLA — integrated phase unwrapping and background removal
///
/// Takes wrapped phase (radians) and returns tissue phase + mask.
/// No Hz→ppm conversion needed (operates in phase domain).
#[wasm_bindgen]
pub fn harperella_wasm_with_progress(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    max_iter: usize,
    tol: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM HARPERELLA: {}x{}x{}, radius={:.1}, max_iter={}", nx, ny, nz, radius, max_iter);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::bgremove::HarperellaParams { radius, max_iter, tol };
    let callback = progress_callback.clone();
    let (tissue_phase, out_mask) = qsm_core::bgremove::harperella(
        phase, mask, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result = tissue_phase;
    result.extend(out_mask.iter().map(|&m| m as f64));

    console_log!("WASM HARPERELLA complete");
    result
}

/// iHARPERELLA — improved integrated phase unwrapping and background removal
///
/// Takes wrapped phase (radians) and returns tissue phase + mask.
/// No Hz→ppm conversion needed (operates in phase domain).
#[wasm_bindgen]
pub fn iharperella_wasm_with_progress(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    max_iter: usize,
    tol: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM iHARPERELLA: {}x{}x{}, radius={:.1}, max_iter={}", nx, ny, nz, radius, max_iter);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::bgremove::HarperellaParams { radius, max_iter, tol };
    let callback = progress_callback.clone();
    let (tissue_phase, out_mask) = qsm_core::bgremove::iharperella(
        phase, mask, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mut result = tissue_phase;
    result.extend(out_mask.iter().map(|&m| m as f64));

    console_log!("WASM iHARPERELLA complete");
    result
}

// ============================================================================
// WASM Exports: Utilities
// ============================================================================

/// Check if WASM module is loaded and working
#[wasm_bindgen]
pub fn wasm_health_check() -> bool {
    console_log!("QSM-WASM module loaded successfully!");
    true
}

/// Get version string
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get dipole kernel for visualization/debugging
#[wasm_bindgen]
pub fn get_dipole_kernel(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
) -> Vec<f64> {
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    qsm_core::kernels::dipole::dipole_kernel(&grid, (bx, by, bz))
}

// ============================================================================
// WASM Exports: NIfTI I/O
// ============================================================================

/// Load a 3D NIfTI file from bytes
///
/// Returns a JS object with: data (Float64Array), dims (array), voxelSize (array), affine (array)
#[wasm_bindgen]
pub fn load_nifti_wasm(bytes: &[u8]) -> Result<js_sys::Object, JsValue> {
    let nifti_data = qsm_core::io::load_nifti(bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Object::new();

    // Data as Float64Array
    let data = js_sys::Float64Array::from(nifti_data.data.as_slice());
    js_sys::Reflect::set(&result, &"data".into(), &data)?;

    // Dimensions
    let dims = js_sys::Array::new();
    dims.push(&JsValue::from(nifti_data.dims.0 as u32));
    dims.push(&JsValue::from(nifti_data.dims.1 as u32));
    dims.push(&JsValue::from(nifti_data.dims.2 as u32));
    js_sys::Reflect::set(&result, &"dims".into(), &dims)?;

    // Voxel size
    let voxel_size = js_sys::Array::new();
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.0));
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.1));
    voxel_size.push(&JsValue::from(nifti_data.voxel_size.2));
    js_sys::Reflect::set(&result, &"voxelSize".into(), &voxel_size)?;

    // Affine matrix
    let affine = js_sys::Float64Array::from(nifti_data.affine.as_slice());
    js_sys::Reflect::set(&result, &"affine".into(), &affine)?;

    console_log!("WASM load_nifti: {}x{}x{}, voxel=({:.2},{:.2},{:.2})",
                 nifti_data.dims.0, nifti_data.dims.1, nifti_data.dims.2,
                 nifti_data.voxel_size.0, nifti_data.voxel_size.1, nifti_data.voxel_size.2);

    Ok(result)
}

/// Load a 4D NIfTI file from bytes (for multi-echo data)
///
/// Returns a JS object with: data (Float64Array), dims (array of 4), voxelSize (array), affine (array)
#[wasm_bindgen]
pub fn load_nifti_4d_wasm(bytes: &[u8]) -> Result<js_sys::Object, JsValue> {
    let (data, dims, voxel_size, affine) = qsm_core::io::load_nifti_4d(bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let result = js_sys::Object::new();

    // Data as Float64Array
    let data_arr = js_sys::Float64Array::from(data.as_slice());
    js_sys::Reflect::set(&result, &"data".into(), &data_arr)?;

    // Dimensions (4D)
    let dims_arr = js_sys::Array::new();
    dims_arr.push(&JsValue::from(dims.0 as u32));
    dims_arr.push(&JsValue::from(dims.1 as u32));
    dims_arr.push(&JsValue::from(dims.2 as u32));
    dims_arr.push(&JsValue::from(dims.3 as u32));
    js_sys::Reflect::set(&result, &"dims".into(), &dims_arr)?;

    // Voxel size
    let voxel_size_arr = js_sys::Array::new();
    voxel_size_arr.push(&JsValue::from(voxel_size.0));
    voxel_size_arr.push(&JsValue::from(voxel_size.1));
    voxel_size_arr.push(&JsValue::from(voxel_size.2));
    js_sys::Reflect::set(&result, &"voxelSize".into(), &voxel_size_arr)?;

    // Affine matrix
    let affine_arr = js_sys::Float64Array::from(affine.as_slice());
    js_sys::Reflect::set(&result, &"affine".into(), &affine_arr)?;

    console_log!("WASM load_nifti_4d: {}x{}x{}x{}", dims.0, dims.1, dims.2, dims.3);

    Ok(result)
}

/// Save data as NIfTI bytes
///
/// # Arguments
/// * `data` - Volume data as Float64Array (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `affine` - 4x4 affine matrix (16 elements, row-major)
///
/// # Returns
/// NIfTI file as Uint8Array
#[wasm_bindgen]
pub fn save_nifti_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    affine: &[f64],
) -> Result<Vec<u8>, JsValue> {
    if affine.len() != 16 {
        return Err(JsValue::from_str("Affine matrix must have 16 elements"));
    }

    let mut affine_arr = [0.0f64; 16];
    affine_arr.copy_from_slice(affine);

    let bytes = qsm_core::io::save_nifti(data, (nx, ny, nz), (vsx, vsy, vsz), &affine_arr)
        .map_err(|e| JsValue::from_str(&e))?;

    console_log!("WASM save_nifti: {}x{}x{}, {} bytes", nx, ny, nz, bytes.len());

    Ok(bytes)
}

/// Save data as gzipped NIfTI bytes (.nii.gz)
#[wasm_bindgen]
pub fn save_nifti_gz_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    affine: &[f64],
) -> Result<Vec<u8>, JsValue> {
    if affine.len() != 16 {
        return Err(JsValue::from_str("Affine matrix must have 16 elements"));
    }

    let mut affine_arr = [0.0f64; 16];
    affine_arr.copy_from_slice(affine);

    let bytes = qsm_core::io::save_nifti_gz(data, (nx, ny, nz), (vsx, vsy, vsz), &affine_arr)
        .map_err(|e| JsValue::from_str(&e))?;

    console_log!("WASM save_nifti_gz: {}x{}x{}, {} bytes (compressed)", nx, ny, nz, bytes.len());

    Ok(bytes)
}

// ============================================================================
// WASM Exports: Brain Extraction (BET)
// ============================================================================

/// BET brain extraction (aligned with FSL-BET2)
///
/// # Arguments
/// * `data` - 3D magnitude image (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain)
/// * `smoothness_factor` - Smoothness constraint (default 1.0, larger = smoother surface)
/// * `gradient_threshold` - Z-gradient for threshold (-1 to 1, positive = larger brain at bottom)
/// * `iterations` - Number of surface evolution iterations
/// * `subdivisions` - Icosphere subdivision level (4 = 2562 vertices)
///
/// # Returns
/// Binary mask as Uint8Array (1 = brain, 0 = background)
#[wasm_bindgen]
pub fn bet_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
) -> Vec<u8> {
    console_log!("WASM BET: {}x{}x{}, fi={:.2}, smooth={:.2}, grad={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::bet::BetParams {
        fractional_intensity,
        smoothness: smoothness_factor,
        gradient_threshold,
        iterations,
        subdivisions,
    };
    let mask = qsm_core::bet::run_bet(data, &grid, &params, |_, _| {});

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    let coverage = 100.0 * mask_count as f64 / mask.len() as f64;
    console_log!("WASM BET complete: {} voxels ({:.1}%)", mask_count, coverage);

    mask
}

/// Run BET with progress callback (aligned with FSL-BET2)
///
/// The callback receives (current_iteration, total_iterations)
#[wasm_bindgen]
pub fn bet_wasm_with_progress(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    fractional_intensity: f64,
    smoothness_factor: f64,
    gradient_threshold: f64,
    iterations: usize,
    subdivisions: usize,
    progress_callback: &js_sys::Function,
) -> Vec<u8> {
    console_log!("WASM BET with progress: {}x{}x{}, fi={:.2}, smooth={:.2}, grad={:.2}, iter={}, subdiv={}",
                 nx, ny, nz, fractional_intensity, smoothness_factor, gradient_threshold, iterations, subdivisions);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let callback = progress_callback.clone();
    let params = qsm_core::bet::BetParams {
        fractional_intensity,
        smoothness: smoothness_factor,
        gradient_threshold,
        iterations,
        subdivisions,
    };
    let mask = qsm_core::bet::run_bet(
        data, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    let coverage = 100.0 * mask_count as f64 / mask.len() as f64;
    console_log!("WASM BET complete: {} voxels ({:.1}%)", mask_count, coverage);

    mask
}

/// Create a simple spherical mask for testing (bypasses BET algorithm)
#[wasm_bindgen]
pub fn create_sphere_mask(
    nx: usize, ny: usize, nz: usize,
    center_x: f64, center_y: f64, center_z: f64,
    radius: f64,
) -> Vec<u8> {
    console_log!("Creating sphere mask: {}x{}x{}, center=({:.1},{:.1},{:.1}), r={:.1}",
                 nx, ny, nz, center_x, center_y, center_z, radius);

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let mask = qsm_core::utils::mask::create_sphere_mask(
        &grid, center_x, center_y, center_z, radius
    );

    let count: usize = mask.iter().map(|&m| m as usize).sum();
    console_log!("Sphere mask: {} voxels ({:.1}%)", count, 100.0 * count as f64 / mask.len() as f64);

    mask
}

/// Otsu's method for automatic thresholding
#[wasm_bindgen]
pub fn otsu_threshold_wasm(data: &[f64], num_bins: usize) -> Vec<u8> {
    console_log!("WASM Otsu: {} voxels, {} bins", data.len(), num_bins);

    let threshold = qsm_core::utils::threshold::otsu_threshold(data, num_bins);
    console_log!("Otsu threshold: {:.4}", threshold);

    // Create binary mask
    let mut mask = vec![0u8; data.len()];
    let mut count = 0usize;
    for (i, &v) in data.iter().enumerate() {
        if v > threshold {
            mask[i] = 1;
            count += 1;
        }
    }

    let coverage = 100.0 * count as f64 / data.len() as f64;
    console_log!("Otsu mask: {} voxels ({:.1}%)", count, coverage);

    mask
}

// ============================================================================
// WASM Exports: Multi-Echo Processing (MCPC-3D-S)
// ============================================================================

/// 3D Gaussian smoothing for phase data (handles wrapping)
///
/// Smooths phase by converting to complex representation, smoothing real/imag
/// separately, then converting back to phase. This correctly handles phase wrapping.
///
/// # Arguments
/// * `phase` - Phase data in radians (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma in voxels
///
/// # Returns
/// Smoothed phase data
#[wasm_bindgen]
pub fn gaussian_smooth_3d_phase_wasm(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
) -> Vec<f64> {
    console_log!("WASM gaussian_smooth_3d_phase: {}x{}x{}, sigma=({:.1},{:.1},{:.1})",
                 nx, ny, nz, sigma_x, sigma_y, sigma_z);

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::utils::multi_echo::gaussian_smooth_3d_phase(
        phase, [sigma_x, sigma_y, sigma_z], mask, &grid
    );

    console_log!("WASM gaussian_smooth_3d_phase complete");
    result
}

/// Hermitian Inner Product (HIP) between two echoes
///
/// Computes HIP = conj(echo1) * echo2 = mag1 * mag2 * exp(i * (phase2 - phase1))
///
/// # Arguments
/// * `phase1`, `mag1` - First echo phase and magnitude
/// * `phase2`, `mag2` - Second echo phase and magnitude
/// * `mask` - Binary mask (nx * ny * nz)
/// * `n` - Total number of voxels
///
/// # Returns
/// Flattened [hip_phase, hip_mag] - first n elements are phase diff, next n are combined mag
#[wasm_bindgen]
pub fn hermitian_inner_product_wasm(
    phase1: &[f64], mag1: &[f64],
    phase2: &[f64], mag2: &[f64],
    mask: &[u8],
    n: usize,
) -> Vec<f64> {
    console_log!("WASM hermitian_inner_product: n={}", n);

    let (hip_phase, hip_mag) = qsm_core::utils::multi_echo::hermitian_inner_product(
        phase1, mag1, phase2, mag2, mask, n
    );

    // Combine into single output
    let mut result = hip_phase;
    result.extend(hip_mag);

    console_log!("WASM hermitian_inner_product complete");
    result
}

/// MCPC-3D-S phase offset estimation for single-coil multi-echo data
///
/// Estimates and removes the phase offset from each echo using the
/// MCPC-3D-S algorithm from MriResearchTools.jl
///
/// # Arguments
/// * `phases_flat` - Flattened phase data [echo0, echo1, ...], each echo is nx*ny*nz
/// * `mags_flat` - Flattened magnitude data [echo0, echo1, ...], each echo is nx*ny*nz
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
/// * `echo1`, `echo2` - Which echoes to use for HIP (0-indexed)
///
/// # Returns
/// Flattened [corrected_phases..., phase_offset]
/// - First n_echoes * n_total elements are corrected phases
/// - Last n_total elements are the estimated phase offset
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mcpc3ds_single_coil_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
    echo1: usize, echo2: usize,
) -> Vec<f64> {
    let n_echoes = tes.len();
    let n_total = nx * ny * nz;

    console_log!("WASM mcpc3ds_single_coil: {}x{}x{}, {} echoes, sigma=({:.1},{:.1},{:.1})",
                 nx, ny, nz, n_echoes, sigma_x, sigma_y, sigma_z);

    // Use slices into the flat input instead of cloning
    let phases: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &phases_flat[e * n_total..(e + 1) * n_total])
        .collect();
    let mags: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
        .collect();

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let (corrected_phases, phase_offset) = qsm_core::utils::multi_echo::phase_offset_removal(
        &phases, &mags, tes, mask,
        [sigma_x, sigma_y, sigma_z], [echo1, echo2],
        qsm_core::unwrap::UnwrapMethod::Romeo,
        &grid,
    );

    // Flatten output: all corrected phases followed by phase_offset
    let mut result = Vec::with_capacity((n_echoes + 1) * n_total);
    for phase in &corrected_phases {
        result.extend(phase);
    }
    result.extend(phase_offset);

    console_log!("WASM mcpc3ds_single_coil complete");
    result
}

/// Calculate B0 field from unwrapped phase using weighted averaging
///
/// Implements calculateB0_unwrapped from MriResearchTools.jl
/// Formula: B0 = (1000 / 2pi) * sum(phase / TE * weight) / sum(weight)
///
/// # Arguments
/// * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `weight_type` - Weighting type: "phase_snr", "phase_var", "average", "tes", "mag"
/// * `n_total` - Number of voxels per echo
///
/// # Returns
/// B0 field in Hz
#[wasm_bindgen]
pub fn calculate_b0_weighted_wasm(
    unwrapped_phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    weight_type: &str,
    n_total: usize,
) -> Vec<f64> {
    let n_echoes = tes.len();

    console_log!("WASM calculate_b0_weighted: {} echoes, {} voxels, type={}",
                 n_echoes, n_total, weight_type);

    // Split flat arrays into per-echo vectors
    let unwrapped_phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| unwrapped_phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| mags_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();

    let wt = qsm_core::utils::multi_echo::B0WeightType::from_str(weight_type);

    // calculate_b0_weighted needs a Grid; infer dims from n_total (treated as 1D for this API)
    let grid = qsm_core::Grid::new(n_total, 1, 1, 1.0, 1.0, 1.0);
    let b0 = qsm_core::utils::multi_echo::calculate_b0_weighted(
        &unwrapped_phases, &mags, tes, mask, wt, &grid
    );

    console_log!("WASM calculate_b0_weighted complete");
    b0
}

/// Full MCPC-3D-S + B0 calculation pipeline
///
/// Combines phase offset removal with weighted B0 calculation.
/// This is the main entry point for multi-echo B0 mapping.
///
/// # Arguments
/// * `phases_flat` - Flattened wrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in ms
/// * `mask` - Binary mask
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma_x`, `sigma_y`, `sigma_z` - Smoothing sigma for phase offset
/// * `weight_type` - B0 weighting type
///
/// # Returns
/// Flattened [b0, phase_offset, corrected_phases...]
/// - First n_total elements: B0 in Hz
/// - Next n_total elements: phase offset
/// - Remaining n_echoes * n_total elements: corrected phases
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mcpc3ds_b0_pipeline_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
    weight_type: &str,
    do_bipolar_correction: bool,
    unwrap_method: &str,
    romeo_individual: bool,
    romeo_correct_global: bool,
) -> Vec<f64> {
    let n_echoes = tes.len();
    let n_total = nx * ny * nz;

    console_log!("WASM field_mapping: {}x{}x{}, {} echoes, unwrap={}, individual={}, correct_global={}, weight={}, bipolar={}",
                 nx, ny, nz, n_echoes, unwrap_method, romeo_individual, romeo_correct_global, weight_type, do_bipolar_correction);

    // Use slices into the flat input instead of cloning (~990 MB savings for large data)
    let phases: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &phases_flat[e * n_total..(e + 1) * n_total])
        .collect();
    let mags: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
        .collect();

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let wt = qsm_core::utils::multi_echo::B0WeightType::from_str(weight_type);

    // Step 1: Phase offset removal (always uses ROMEO internally for HIP unwrapping)
    let (mut corrected_phases, phase_offset) = qsm_core::utils::multi_echo::phase_offset_removal(
        &phases, &mags, tes, mask,
        [sigma_x, sigma_y, sigma_z], [0, 1],
        qsm_core::unwrap::UnwrapMethod::Romeo,
        &grid,
    );

    // Step 2: Bipolar correction (after offset removal, before unwrapping — matches MriResearchTools.jl)
    if do_bipolar_correction && n_echoes >= 3 {
        console_log!("WASM bipolar correction (post-offset-removal)");
        let mag_refs: Vec<&[f64]> = (0..n_echoes)
            .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
            .collect();
        qsm_core::utils::multi_echo::bipolar_correction(
            &mut corrected_phases, &mag_refs, tes, mask,
            [sigma_x, sigma_y, sigma_z], &grid,
        );
    }

    // Step 3: Multi-echo unwrapping (user-selected method)
    let mag_refs: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
        .collect();
    let unwrapped: Vec<Vec<f64>> = match unwrap_method {
        "laplacian" => {
            // Per-echo Laplacian unwrapping (matching Julia's laplacian_combine)
            // No inter-echo alignment — Laplacian removes harmonic component
            // independently per echo, and calculate_b0_weighted handles
            // the per-echo phase/TE division and averaging
            corrected_phases.iter()
                .map(|phase| qsm_core::unwrap::laplacian_unwrap(phase, mask, &grid))
                .collect()
        }
        _ => {
            let params = qsm_core::unwrap::romeo::RomeoParams {
                individual: romeo_individual,
                correct_global: romeo_correct_global,
                ..Default::default()
            };
            qsm_core::unwrap::romeo::unwrap_romeo_multi_echo(
                &corrected_phases, &mag_refs, tes, mask,
                &params, &grid,
            )
        }
    };

    // Step 4: Weighted B0 averaging
    let b0 = qsm_core::utils::multi_echo::calculate_b0_weighted(
        &unwrapped, &mags, tes, mask, wt, &grid,
    );

    // Flatten output: b0, phase_offset, then all corrected phases
    let mut result = Vec::with_capacity((2 + n_echoes) * n_total);
    result.extend(b0);
    result.extend(phase_offset);
    for phase in &corrected_phases {
        result.extend(phase);
    }

    console_log!("WASM mcpc3ds_b0_pipeline complete");
    result
}

/// Multi-echo linear fit with magnitude weighting
///
/// Fits a linear model: phase = slope * TE + intercept
/// using weighted least squares with magnitude as weights.
///
/// # Arguments
/// * `unwrapped_phases_flat` - Flattened unwrapped phases [echo0, echo1, ...]
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `tes` - Echo times in seconds
/// * `mask` - Binary mask
/// * `n_total` - Voxels per echo
/// * `estimate_offset` - If true, estimate phase offset (intercept)
/// * `reliability_percentile` - Percentile for reliability masking (0-100, 0=disable)
///
/// # Returns
/// Flattened [field_hz, phase_offset, fit_residual, reliability_mask]
/// - First n_total: field in Hz
/// - Next n_total: phase offset in radians
/// - Next n_total: fit residual
/// - Next n_total: reliability mask (as f64, 0 or 1)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn multi_echo_linear_fit_wasm(
    unwrapped_phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    n_total: usize,
    estimate_offset: bool,
    reliability_percentile: f64,
) -> Vec<f64> {
    let n_echoes = tes.len();

    console_log!("WASM multi_echo_linear_fit: {} echoes, {} voxels, offset={}, reliability={}%",
                 n_echoes, n_total, estimate_offset, reliability_percentile);

    // Use slices into the flat input instead of cloning
    let unwrapped_phases: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &unwrapped_phases_flat[e * n_total..(e + 1) * n_total])
        .collect();
    let mags: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
        .collect();

    let result = qsm_core::utils::multi_echo::multi_echo_linear_fit(
        &unwrapped_phases, &mags, tes, mask,
        estimate_offset, reliability_percentile
    );

    // Convert field to Hz
    let field_hz = qsm_core::utils::multi_echo::field_to_hz(&result.field);

    // Flatten output
    let mut output = Vec::with_capacity(4 * n_total);
    output.extend(field_hz);
    output.extend(result.phase_offset);
    output.extend(result.fit_residual);
    output.extend(result.reliability_mask.iter().map(|&v| v as f64));

    console_log!("WASM multi_echo_linear_fit complete");
    output
}


/// Bipolar gradient correction for multi-echo phase data
///
/// Removes linear phase artefact caused by bipolar readout gradients.
/// Requires at least 3 echoes; with fewer, returns input unchanged.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn bipolar_correction_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    tes: &[f64],
    mask: &[u8],
    sigma_x: f64, sigma_y: f64, sigma_z: f64,
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_echoes = tes.len();
    let n_total = nx * ny * nz;

    console_log!("WASM bipolar_correction: {} echoes, {}x{}x{}, sigma=[{:.1},{:.1},{:.1}]",
                 n_echoes, nx, ny, nz, sigma_x, sigma_y, sigma_z);

    let mut phases: Vec<Vec<f64>> = (0..n_echoes)
        .map(|e| phases_flat[e * n_total..(e + 1) * n_total].to_vec())
        .collect();
    let mags: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &mags_flat[e * n_total..(e + 1) * n_total])
        .collect();

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    qsm_core::utils::multi_echo::bipolar_correction(
        &mut phases, &mags, tes, mask,
        [sigma_x, sigma_y, sigma_z], &grid,
    );

    let mut output = Vec::with_capacity(n_echoes * n_total);
    for echo in &phases {
        output.extend_from_slice(echo);
    }

    console_log!("WASM bipolar_correction complete");
    output
}

// ============================================================================
// Bias Correction Functions
// ============================================================================

/// Bias field correction (makehomogeneous)
///
/// Corrects RF receive field inhomogeneities in magnitude images using
/// the boxsegment approach from MriResearchTools.jl.
///
/// # Arguments
/// * `mag` - Magnitude data (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `sigma_mm` - Smoothing sigma in mm (will be clamped to 10% of FOV)
/// * `nbox` - Number of boxes per dimension for segmentation
///
/// # Returns
/// Bias-corrected magnitude
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn makehomogeneous_wasm(
    mag: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    sigma_mm: f64,
    nbox: usize,
) -> Vec<f64> {
    console_log!("WASM makehomogeneous: {}x{}x{}, voxel=[{:.2},{:.2},{:.2}]mm, sigma={:.1}mm, nbox={}",
                 nx, ny, nz, vsx, vsy, vsz, sigma_mm, nbox);

    // Clamp sigma to 10% of minimum FOV dimension
    let fov_min = (nx as f64 * vsx).min(ny as f64 * vsy).min(nz as f64 * vsz);
    let sigma_clamped = sigma_mm.min(fov_min * 0.1);

    if (sigma_clamped - sigma_mm).abs() > 0.1 {
        console_log!("WASM makehomogeneous: sigma clamped from {:.1} to {:.1}mm", sigma_mm, sigma_clamped);
    }

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let result = qsm_core::utils::bias_correction::makehomogeneous(
        mag, &grid, sigma_clamped, nbox
    );

    console_log!("WASM makehomogeneous complete");
    result
}

/// RSS (Root Sum of Squares) magnitude combination
///
/// Combines multi-echo magnitude images using RSS for improved SNR.
///
/// # Arguments
/// * `mags_flat` - Flattened magnitudes [echo0, echo1, ...]
/// * `n_echoes` - Number of echoes
/// * `n_total` - Voxels per echo (nx * ny * nz)
///
/// # Returns
/// RSS-combined magnitude
#[wasm_bindgen]
pub fn rss_combine_wasm(
    mags_flat: &[f64],
    n_echoes: usize,
    n_total: usize,
) -> Vec<f64> {
    console_log!("WASM RSS combine: {} echoes, {} voxels each", n_echoes, n_total);

    let result = qsm_core::utils::bias_correction::rss_combine(mags_flat, n_echoes, n_total);

    console_log!("WASM RSS combine complete");
    result
}

// ============================================================================
// WASM Exports: QSMART Pipeline Functions
// ============================================================================

/// Frangi vesselness filter for vessel detection
///
/// Detects tubular structures (vessels) using multi-scale Hessian eigenvalue analysis.
///
/// # Arguments
/// * `data` - Input 3D volume (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `scale_min` - Minimum sigma for multi-scale analysis (default 0.5)
/// * `scale_max` - Maximum sigma (default 6.0)
/// * `scale_ratio` - Step between scales (default 0.5)
/// * `alpha` - Plate vs line sensitivity (default 0.5)
/// * `beta` - Blob vs line sensitivity (default 0.5)
/// * `c` - Noise threshold (default 500)
/// * `black_white` - Detect dark vessels (true) or bright (false)
///
/// # Returns
/// Vesselness response (0-1)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn frangi_filter_3d_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    scale_min: f64, scale_max: f64, scale_ratio: f64,
    alpha: f64, beta: f64, c: f64,
    black_white: bool,
) -> Vec<f64> {
    console_log!("WASM Frangi: {}x{}x{}, scales=[{:.1},{:.1}], c={}",
                 nx, ny, nz, scale_min, scale_max, c);

    let params = qsm_core::utils::frangi::FrangiParams {
        scale_range: [scale_min, scale_max],
        scale_ratio,
        alpha,
        beta,
        c,
        black_white,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::utils::frangi::frangi_filter_3d(data, &grid, &params, |_, _| {});

    console_log!("WASM Frangi complete");
    result.vesselness
}

/// Frangi filter with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn frangi_filter_3d_wasm_with_progress(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    scale_min: f64, scale_max: f64, scale_ratio: f64,
    alpha: f64, beta: f64, c: f64,
    black_white: bool,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM Frangi with progress: {}x{}x{}", nx, ny, nz);

    let params = qsm_core::utils::frangi::FrangiParams {
        scale_range: [scale_min, scale_max],
        scale_ratio,
        alpha,
        beta,
        c,
        black_white,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let callback = progress_callback.clone();
    let result = qsm_core::utils::frangi::frangi_filter_3d(
        data, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM Frangi complete");
    result.vesselness
}

/// Generate vasculature mask for QSMART
///
/// Uses bottom-hat filtering and Frangi vesselness to detect blood vessels.
///
/// # Arguments
/// * `magnitude` - Average magnitude image (ideally bias-corrected)
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sphere_radius` - Radius for bottom-hat filter (default 8)
/// * `frangi_scale_min`, `frangi_scale_max` - Frangi scale range (default [0.5, 6])
/// * `frangi_scale_ratio` - Frangi scale step (default 0.5)
/// * `frangi_c` - Frangi C parameter (default 500)
///
/// # Returns
/// Complementary mask (1 = tissue, 0 = vessel)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn vasculature_mask_wasm(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sphere_radius: i32,
    frangi_scale_min: f64, frangi_scale_max: f64, frangi_scale_ratio: f64,
    frangi_c: f64,
) -> Vec<f64> {
    console_log!("WASM vasculature_mask: {}x{}x{}, sphere_r={}, frangi_c={}",
                 nx, ny, nz, sphere_radius, frangi_c);

    let params = qsm_core::utils::vasculature::VasculatureParams {
        sphere_radius,
        frangi_scale_range: [frangi_scale_min, frangi_scale_max],
        frangi_scale_ratio,
        frangi_c,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::utils::vasculature::generate_vasculature_mask(magnitude, mask, &grid, &params, |_, _| {});

    console_log!("WASM vasculature_mask complete");
    result
}

/// Vasculature mask with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn vasculature_mask_wasm_with_progress(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    sphere_radius: i32,
    frangi_scale_min: f64, frangi_scale_max: f64, frangi_scale_ratio: f64,
    frangi_c: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM vasculature_mask with progress: {}x{}x{}", nx, ny, nz);

    let params = qsm_core::utils::vasculature::VasculatureParams {
        sphere_radius,
        frangi_scale_range: [frangi_scale_min, frangi_scale_max],
        frangi_scale_ratio,
        frangi_c,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let callback = progress_callback.clone();
    let result = qsm_core::utils::vasculature::generate_vasculature_mask(
        magnitude, mask, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM vasculature_mask complete");
    result
}

/// SDF (Spatially Dependent Filtering) background field removal for QSMART
///
/// Variable-radius Gaussian filtering where kernel size depends on proximity to boundary.
///
/// # Arguments
/// * `tfs` - Total field shift (weighted by mask if using R_0)
/// * `mask` - Weighted mask (mask * R_0 for reliability weighting)
/// * `vasc_only` - Vasculature mask (1 = tissue, 0 = vessel). Use all-ones for stage 1.
/// * `nx`, `ny`, `nz` - Dimensions
/// * `sigma1` - Primary smoothing sigma (10 for stage1, 8 for stage2)
/// * `sigma2` - Vasculature proximity sigma (0 for stage1, 2 for stage2)
/// * `lower_lim` - Proximity clamping value (default 0.6)
/// * `curv_constant` - Curvature scaling (default 500)
/// * `use_curvature` - Enable curvature-based weighting
///
/// # Returns
/// Local field shift (background removed)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sdf_wasm(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma1: f64, sigma2: f64,
    lower_lim: f64, curv_constant: f64,
    use_curvature: bool,
) -> Vec<f64> {
    console_log!("WASM SDF: {}x{}x{}, sigma1={}, sigma2={}, curv={}",
                 nx, ny, nz, sigma1, sigma2, use_curvature);

    let params = qsm_core::bgremove::sdf::SdfParams {
        sigma1,
        sigma2,
        spatial_radius: 8,
        lower_lim,
        curv_constant,
        use_curvature,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::bgremove::sdf::sdf(tfs, mask, vasc_only, &grid, &params, |_, _| {});

    console_log!("WASM SDF complete");
    result
}

/// SDF with progress callback
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sdf_wasm_with_progress(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma1: f64, sigma2: f64,
    spatial_radius: i32,
    lower_lim: f64, curv_constant: f64,
    use_curvature: bool,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    console_log!("WASM SDF with progress: {}x{}x{}", nx, ny, nz);

    let params = qsm_core::bgremove::sdf::SdfParams {
        sigma1,
        sigma2,
        spatial_radius,
        lower_lim,
        curv_constant,
        use_curvature,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let callback = progress_callback.clone();
    let result = qsm_core::bgremove::sdf::sdf(
        tfs, mask, vasc_only, &grid, &params,
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(&this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32));
        }
    );

    console_log!("WASM SDF complete");
    result
}

/// QSMART offset adjustment
///
/// Combines two-stage QSM results with offset adjustment for consistency.
///
/// # Arguments
/// * `removed_voxels` - Voxels in stage 1 but not stage 2 (mask*R_0 - vasc_only)
/// * `lfs_sdf` - Local field from stage 1 (in ppm)
/// * `chi_1` - Susceptibility from stage 1 (whole ROI)
/// * `chi_2` - Susceptibility from stage 2 (tissue only)
/// * `nx`, `ny`, `nz` - Dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bx`, `by`, `bz` - B0 field direction
/// * `ppm` - PPM conversion factor
///
/// # Returns
/// Combined and offset-adjusted susceptibility map
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn qsmart_adjust_offset_wasm(
    removed_voxels: &[f64],
    lfs_sdf: &[f64],
    chi_1: &[f64],
    chi_2: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    ppm: f64,
) -> Vec<f64> {
    console_log!("WASM QSMART offset adjustment: {}x{}x{}", nx, ny, nz);

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let result = qsm_core::utils::qsmart::adjust_offset(
        removed_voxels, lfs_sdf, chi_1, chi_2,
        &grid, (bx, by, bz), ppm
    );

    console_log!("WASM QSMART offset adjustment complete");
    result
}

/// Calculate Gaussian curvature at mask boundary
///
/// Used for curvature-based edge weighting in QSMART SDF.
///
/// # Arguments
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Dimensions
///
/// # Returns
/// Flattened [gaussian_curvature, mean_curvature] - each n_total elements
#[wasm_bindgen]
pub fn curvature_wasm(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    console_log!("WASM curvature: {}x{}x{}", nx, ny, nz);

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::utils::curvature::calculate_gaussian_curvature(mask, &grid);

    // Combine outputs
    let mut output = result.gaussian_curvature;
    output.extend(result.mean_curvature);

    console_log!("WASM curvature complete: {} surface voxels", result.surface_indices.len());
    output
}

// ============================================================================
// WASM Exports: Susceptibility Weighted Imaging (SWI)
// ============================================================================

/// Calculate SWI from unwrapped phase and magnitude
///
/// Pipeline: high-pass filter phase → create phase mask → multiply with magnitude.
///
/// # Arguments
/// * `phase` - Unwrapped phase (nx * ny * nz)
/// * `magnitude` - Magnitude image (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `hp_sigma_x`, `hp_sigma_y`, `hp_sigma_z` - High-pass filter sigma in voxels
/// * `scaling_type` - Phase scaling: 0=Tanh, 1=NegativeTanh, 2=Positive, 3=Negative, 4=Triangular
/// * `strength` - Phase scaling strength
///
/// # Returns
/// SWI image (magnitude × phase mask)
#[wasm_bindgen]
pub fn calculate_swi_wasm(
    phase: &[f64],
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    hp_sigma_x: f64, hp_sigma_y: f64, hp_sigma_z: f64,
    scaling_type: u8,
    strength: f64,
) -> Vec<f64> {
    console_log!("WASM SWI: {}x{}x{}, sigma=({},{},{}), scaling={}, strength={}",
                 nx, ny, nz, hp_sigma_x, hp_sigma_y, hp_sigma_z, scaling_type, strength);

    let scaling = match scaling_type {
        0 => qsm_core::swi::PhaseScaling::Tanh,
        1 => qsm_core::swi::PhaseScaling::NegativeTanh,
        2 => qsm_core::swi::PhaseScaling::Positive,
        3 => qsm_core::swi::PhaseScaling::Negative,
        _ => qsm_core::swi::PhaseScaling::Triangular,
    };

    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let swi_params = qsm_core::swi::SwiParams {
        hp_sigma: [hp_sigma_x, hp_sigma_y, hp_sigma_z],
        scaling,
        strength,
        ..Default::default()
    };
    let result = qsm_core::swi::calculate_swi(
        phase, magnitude, mask, &grid, &swi_params,
    );

    console_log!("WASM SWI complete");
    result
}

/// Minimum intensity projection along the z-axis
///
/// For each (x, y) position, takes the minimum value over a sliding window
/// of `window` slices along z.
///
/// # Arguments
/// * `data` - 3D volume (nx * ny * nz, Fortran order)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `window` - Number of slices in the projection window
///
/// # Returns
/// MIP volume with dimensions nx × ny × (nz - window + 1)
#[wasm_bindgen]
pub fn create_mip_wasm(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    window: usize,
) -> Vec<f64> {
    console_log!("WASM MIP: {}x{}x{}, window={}", nx, ny, nz, window);

    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let result = qsm_core::swi::create_mip(data, &grid, window);

    console_log!("WASM MIP complete: output nz={}", if window <= nz { nz - window + 1 } else { 0 });
    result
}

// ============================================================================
// R2* / T2* Mapping
// ============================================================================

/// R2* mapping using ARLO algorithm
///
/// # Arguments
/// * `magnitude` - Interleaved multi-echo magnitude data (n_voxels × n_echoes)
/// * `mask` - Binary mask (n_voxels)
/// * `echo_times` - Echo times in seconds
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// R2* map (n_voxels). Returns empty vec on error.
#[wasm_bindgen]
pub fn r2star_arlo_wasm(
    magnitude: &[f64],
    mask: &[u8],
    echo_times: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    console_log!("WASM R2* ARLO: {}x{}x{}, {} echoes", nx, ny, nz, echo_times.len());
    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let (r2star, _s0) = qsm_core::utils::r2star_arlo(magnitude, mask, echo_times, &grid);
    console_log!("WASM R2* complete");
    r2star
}

// ============================================================================
// Algorithm Default Parameters
// ============================================================================

// ============================================================================
// FANSI-family dipole inversions (NDI, FANSI nlTV/nlTGV, L1-QSM, WH-QSM, HD-QSM)
//
// Same convention as the other inversions: Hz->ppm normalize the input field via
// hz_to_ppm_scale(field_strength), run in ppm (phase_scale stays at the qsm-core
// default of 1.0), then convert the result back. Only the `_with_progress` variant
// is exposed (the worker always uses it).
// ============================================================================

/// NDI (Nonlinear Dipole Inversion) with progress callback.
#[wasm_bindgen]
pub fn ndi_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    tau: f64,
    alpha: f64,
    max_iter: usize,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM NDI: {}x{}x{}, tau={:.3}, alpha={:.2e}, max_iter={}", nx, ny, nz, tau, alpha, max_iter);
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::NdiParams { tau, alpha, max_iter, ..Default::default() };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let cb = progress_callback.clone();
    let chi = qsm_core::inversion::ndi(&field_norm, mask, &grid, (bx, by, bz), &params, |c, t| {
        let _ = cb.call2(&JsValue::null(), &JsValue::from(c as u32), &JsValue::from(t as u32));
    });
    chi.iter().map(|&v| v / scale).collect()
}

/// FANSI nonlinear TV / TGV with progress callback (`is_tgv` selects nlTGV).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn fansi_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha1: f64,
    mu1: f64,
    mu2: f64,
    alpha0: f64,
    mu0: f64,
    max_iter: usize,
    tol_update: f64,
    is_tgv: bool,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM FANSI (tgv={}): {}x{}x{}, alpha1={:.2e}, max_iter={}", is_tgv, nx, ny, nz, alpha1, max_iter);
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::FansiParams {
        alpha1, mu1, mu2, alpha0, mu0, max_iter, tol_update, is_tgv, ..Default::default()
    };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let cb = progress_callback.clone();
    let chi = qsm_core::inversion::fansi(&field_norm, mask, &grid, (bx, by, bz), &params, |c, t| {
        let _ = cb.call2(&JsValue::null(), &JsValue::from(c as u32), &JsValue::from(t as u32));
    });
    chi.iter().map(|&v| v / scale).collect()
}

/// L1-QSM (L1 data-fidelity) with progress callback.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn l1qsm_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha1: f64,
    mu1: f64,
    mu2: f64,
    mu3: f64,
    lambda: f64,
    max_iter: usize,
    tol_update: f64,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM L1-QSM: {}x{}x{}, alpha1={:.2e}, lambda={:.3}, max_iter={}", nx, ny, nz, alpha1, lambda, max_iter);
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::L1QsmParams {
        alpha1, mu1, mu2, mu3, lambda, max_iter, tol_update, ..Default::default()
    };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let cb = progress_callback.clone();
    let chi = qsm_core::inversion::l1qsm(&field_norm, mask, &grid, (bx, by, bz), &params, |c, t| {
        let _ = cb.call2(&JsValue::null(), &JsValue::from(c as u32), &JsValue::from(t as u32));
    });
    chi.iter().map(|&v| v / scale).collect()
}

/// WH-QSM (Weak-Harmonic) with progress callback.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn whqsm_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha1: f64,
    mu1: f64,
    mu2: f64,
    beta: f64,
    muh: f64,
    max_iter: usize,
    tol_update: f64,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM WH-QSM: {}x{}x{}, alpha1={:.2e}, beta={}, max_iter={}", nx, ny, nz, alpha1, beta, max_iter);
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::WhQsmParams {
        alpha1, mu1, mu2, beta, muh, max_iter, tol_update, ..Default::default()
    };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let cb = progress_callback.clone();
    let chi = qsm_core::inversion::whqsm(&field_norm, mask, &grid, (bx, by, bz), &params, |c, t| {
        let _ = cb.call2(&JsValue::null(), &JsValue::from(c as u32), &JsValue::from(t as u32));
    });
    chi.iter().map(|&v| v / scale).collect()
}

/// HD-QSM (Hybrid two-stage L1->L2) with progress callback.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn hdqsm_wasm_with_progress(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bx: f64, by: f64, bz: f64,
    alpha_l2: f64,
    mu1_l2: f64,
    mu2: f64,
    max_iter_l1: usize,
    max_iter_l2: usize,
    tol_update: f64,
    field_strength: f64,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let scale = hz_to_ppm_scale(field_strength);
    console_log!("WASM HD-QSM: {}x{}x{}, alphaL2={:.2e}, L1={}, L2={}", nx, ny, nz, alpha_l2, max_iter_l1, max_iter_l2);
    let grid = qsm_core::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = qsm_core::inversion::HdQsmParams {
        alpha_l2, mu1_l2, mu2, max_iter_l1, max_iter_l2, tol_update,
    };
    let field_norm: Vec<f64> = local_field.iter().map(|&v| v * scale).collect();
    let cb = progress_callback.clone();
    let chi = qsm_core::inversion::hdqsm(&field_norm, mask, &grid, (bx, by, bz), &params, |c, t| {
        let _ = cb.call2(&JsValue::null(), &JsValue::from(c as u32), &JsValue::from(t as u32));
    });
    chi.iter().map(|&v| v / scale).collect()
}

// Each get_*_defaults() serializes the matching qsmxt-config config struct, whose
// Default impl sources its values from qsm-core. qsmxt-config is the single source of
// truth for the parameter set (field names + defaults); these bindings just hand it to
// JS as JSON. Adding a field there automatically flows through here — no hand-typed list.
macro_rules! config_defaults {
    ($fn_name:ident, $config:ty) => {
        #[wasm_bindgen]
        pub fn $fn_name() -> String {
            serde_json::to_string(&<$config>::default())
                .expect("config serialization is infallible")
        }
    };
}

config_defaults!(get_rts_defaults, qsmxt_config::config::RtsConfig);
config_defaults!(get_tv_defaults, qsmxt_config::config::TvConfig);
config_defaults!(get_tkd_defaults, qsmxt_config::config::TkdConfig);
config_defaults!(get_tgv_defaults, qsmxt_config::config::TgvConfig);
config_defaults!(get_bet_defaults, qsmxt_config::config::BetConfig);
config_defaults!(get_vsharp_defaults, qsmxt_config::config::VsharpConfig);
config_defaults!(get_pdf_defaults, qsmxt_config::config::PdfConfig);
config_defaults!(get_lbv_defaults, qsmxt_config::config::LbvConfig);
config_defaults!(get_ismv_defaults, qsmxt_config::config::IsmvConfig);
config_defaults!(get_swi_defaults, qsmxt_config::config::SwiConfig);
config_defaults!(get_sharp_defaults, qsmxt_config::config::SharpConfig);
config_defaults!(get_resharp_defaults, qsmxt_config::config::ResharpConfig);
config_defaults!(get_harperella_defaults, qsmxt_config::config::HarperellaConfig);
config_defaults!(get_tikhonov_defaults, qsmxt_config::config::TikhonovConfig);
config_defaults!(get_nltv_defaults, qsmxt_config::config::NltvConfig);
config_defaults!(get_ndi_defaults, qsmxt_config::config::NdiConfig);
config_defaults!(get_fansi_defaults, qsmxt_config::config::FansiConfig);
config_defaults!(get_l1qsm_defaults, qsmxt_config::config::L1qsmConfig);
config_defaults!(get_whqsm_defaults, qsmxt_config::config::WhqsmConfig);
config_defaults!(get_hdqsm_defaults, qsmxt_config::config::HdqsmConfig);
config_defaults!(get_medi_defaults, qsmxt_config::config::MediConfig);
config_defaults!(get_tfi_defaults, qsmxt_config::config::TfiConfig);
config_defaults!(get_qsmart_defaults, qsmxt_config::config::QsmartConfig);
config_defaults!(get_romeo_defaults, qsmxt_config::config::RomeoConfig);
config_defaults!(get_mcpc3ds_defaults, qsmxt_config::config::Mcpc3dsConfig);
config_defaults!(get_linear_fit_defaults, qsmxt_config::config::LinearFitConfig);
config_defaults!(get_homogeneity_defaults, qsmxt_config::config::HomogeneityConfig);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
    }
}

// ============================================================================
// Pipeline Configuration (via qsmxt-config library)
// ============================================================================

// qsmbly builds its config as a JSON object (its UI→schema adapter) and lets
// qsmxt-config do all serialization — so command, methods, the downloadable .toml,
// and the pipeline-execution config are one canonical representation produced by the
// library, never a hand-rolled parallel TOML. `mask_section` is the CLI-style mask
// string parsed here (the tagged mask-op enums round-trip cleanly via serde, unlike a
// hand-written TOML serializer).
fn config_from_json(
    config_json: &str,
    mask_section: &str,
) -> Result<qsmxt_config::PipelineConfig, String> {
    let mut config: qsmxt_config::PipelineConfig =
        serde_json::from_str(config_json).map_err(|e| format!("ERROR: {}", e))?;
    apply_mask_section(&mut config, mask_section);
    Ok(config)
}

/// Serialize a config (JSON, plus CLI-style mask string) to canonical TOML —
/// identical to what the qsmxt.rs CLI writes (all algorithms). Returns "ERROR: ..." on failure.
#[wasm_bindgen]
pub fn config_json_to_toml_wasm(config_json: &str, mask_section: &str) -> String {
    match config_from_json(config_json, mask_section) {
        Ok(config) => config.to_toml().unwrap_or_else(|e| format!("ERROR: {}", e)),
        Err(e) => e,
    }
}

/// Like config_json_to_toml_wasm, but prunes inversion/bg_removal to the selected
/// algorithm only (the omitted ones round-trip as defaults). For the downloadable
/// settings file. Returns "ERROR: ..." on failure.
#[wasm_bindgen]
pub fn config_json_to_toml_selected_wasm(config_json: &str, mask_section: &str) -> String {
    match config_from_json(config_json, mask_section) {
        Ok(config) => config.to_toml_selected().unwrap_or_else(|e| format!("ERROR: {}", e)),
        Err(e) => e,
    }
}

/// Generate a qsmxt CLI command from a config (JSON + mask string).
/// Returns the command string, or an error message prefixed with "ERROR: ".
#[wasm_bindgen]
pub fn generate_command_wasm(config_json: &str, mask_section: &str) -> String {
    match config_from_json(config_json, mask_section) {
        Ok(config) => qsmxt_config::generate_command(&config),
        Err(e) => e,
    }
}

/// Generate a methods section with citations from a config (JSON + mask string).
/// `tool` should be "qsmxt.rs" or "QSMbly" to credit the correct tool.
/// Returns markdown text, or an error message prefixed with "ERROR: ".
#[wasm_bindgen]
pub fn generate_methods_wasm(config_json: &str, tool: &str, mask_section: &str) -> String {
    match config_from_json(config_json, mask_section) {
        Ok(config) => qsmxt_config::methods::generate_methods_for(&config, tool),
        Err(e) => e,
    }
}

/// Parse a CLI-style mask string ("input,gen,refine,...") into the config's mask
/// sections, so command and methods both reflect the real mask. The TOML can't
/// easily carry the tagged mask-op enums, so the UI passes the same string the CLI
/// uses and we parse it here with qsmxt-config's parsers. Empty string = no change.
fn apply_mask_section(config: &mut qsmxt_config::PipelineConfig, mask_section: &str) {
    let parts: Vec<&str> = mask_section
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    // Need an input plus at least one (generator) op.
    if parts.len() < 2 {
        return;
    }
    let Some(input) = qsmxt_config::parse_masking_input(parts[0]) else {
        return;
    };
    let mut ops = Vec::new();
    for p in &parts[1..] {
        match qsmxt_config::parse_mask_op(p) {
            Ok(op) => ops.push(op),
            Err(_) => return, // unrecognised op — leave the default mask untouched
        }
    }
    let generator = ops.remove(0);
    config.masking.sections = vec![qsmxt_config::MaskSection {
        input,
        generator,
        refinements: ops,
    }];
}

/// Return the default PipelineConfig as a TOML string.
#[wasm_bindgen]
pub fn get_default_config_toml_wasm() -> String {
    qsmxt_config::PipelineConfig::default()
        .to_toml()
        .unwrap_or_else(|e| format!("ERROR: {}", e))
}

/// Return the default PipelineConfig as a JSON string.
#[wasm_bindgen]
pub fn get_default_config_json_wasm() -> String {
    qsmxt_config::PipelineConfig::default()
        .to_json()
        .unwrap_or_else(|e| format!("ERROR: {}", e))
}

/// Validate a TOML config string. Returns empty string on success, error message on failure.
#[wasm_bindgen]
pub fn validate_config_wasm(toml_string: &str) -> String {
    match qsmxt_config::PipelineConfig::from_toml(toml_string) {
        Err(e) => format!("Parse error: {}", e),
        Ok(_) => String::new(),
    }
}

// =========================================================================
// Shared pipeline stage functions (guaranteed identical to qsmxt.rs)
// =========================================================================

/// Run field mapping: multi-echo phase → B0 field map (ppm).
///
/// Takes a TOML config string and returns [b0_field_ppm, phase_offset (if any)].
/// Echo times are in seconds.
#[wasm_bindgen]
pub fn run_field_mapping_wasm(
    phases_flat: &[f64],
    mags_flat: &[f64],
    mask: &[u8],
    echo_times: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    field_strength: f64,
    config_toml: &str,
) -> Vec<f64> {
    let n_echoes = echo_times.len();
    let n_total = nx * ny * nz;

    let phases: Vec<&[f64]> = (0..n_echoes)
        .map(|e| &phases_flat[e * n_total..(e + 1) * n_total])
        .collect();
    let mags: Vec<&[f64]> = if mags_flat.is_empty() {
        Vec::new()
    } else {
        (0..n_echoes).map(|e| &mags_flat[e * n_total..(e + 1) * n_total]).collect()
    };
    let mag_opt: Option<&[&[f64]]> = if mags.is_empty() { None } else { Some(&mags) };

    let config = qsmxt_config::PipelineConfig::from_toml(config_toml)
        .unwrap_or_default();
    let (fm_config, _, _, _) = qsmxt_config::to_pipeline_stages(&config);
    let meta = qsmxt_config::to_scan_metadata(
        (nx, ny, nz), (vsx, vsy, vsz), echo_times, field_strength, (0.0, 0.0, 1.0),
    );

    let result = qsm_core::pipeline::run_field_mapping(
        &phases, mag_opt, mask, &meta, &fm_config, &mut |_, _| {},
    );

    match result {
        Ok(r) => {
            let mut out = r.b0_field_ppm;
            if let Some(offset) = r.phase_offset {
                out.extend(offset);
            }
            out
        }
        Err(e) => {
            console_log!("run_field_mapping_wasm error: {}", e);
            vec![0.0; n_total]
        }
    }
}

/// Run background removal: total field → local field (ppm).
///
/// Returns [local_field_ppm, eroded_mask_f64].
#[wasm_bindgen]
pub fn run_bg_removal_wasm(
    field_ppm: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    field_strength: f64,
    config_toml: &str,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let config = qsmxt_config::PipelineConfig::from_toml(config_toml)
        .unwrap_or_default();
    let (_, bg_config, _, _) = qsmxt_config::to_pipeline_stages(&config);
    let meta = qsmxt_config::to_scan_metadata(
        (nx, ny, nz), (vsx, vsy, vsz), &[], field_strength, (0.0, 0.0, 1.0),
    );

    let this = JsValue::null();
    let result = qsm_core::pipeline::run_bg_removal(
        field_ppm, mask, &meta, &bg_config,
        &mut |cur, total| {
            let _ = progress_callback.call2(
                &this, &JsValue::from(cur as u32), &JsValue::from(total as u32),
            );
        },
    );

    match result {
        Ok(r) => {
            let mut out = r.local_field_ppm;
            let mask_f64: Vec<f64> = r.eroded_mask.iter().map(|&m| m as f64).collect();
            out.extend(mask_f64);
            out
        }
        Err(e) => {
            console_log!("run_bg_removal_wasm error: {}", e);
            vec![0.0; n_total * 2]
        }
    }
}

/// Run dipole inversion: local field → susceptibility (ppm).
///
/// Handles MEDI unit conversion internally.
#[wasm_bindgen]
pub fn run_dipole_inversion_wasm(
    local_field_ppm: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    field_strength: f64,
    echo_times: &[f64],
    bx: f64, by: f64, bz: f64,
    magnitude: &[f64],
    config_toml: &str,
    progress_callback: &js_sys::Function,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let config = qsmxt_config::PipelineConfig::from_toml(config_toml)
        .unwrap_or_default();
    let (_, _, inv_config, _) = qsmxt_config::to_pipeline_stages(&config);
    let meta = qsmxt_config::to_scan_metadata(
        (nx, ny, nz), (vsx, vsy, vsz), echo_times, field_strength, (bx, by, bz),
    );

    let mag_opt: Option<&[f64]> = if magnitude.is_empty() { None } else { Some(magnitude) };

    let this = JsValue::null();
    let result = qsm_core::pipeline::run_dipole_inversion(
        local_field_ppm, mask, &meta, &inv_config, mag_opt,
        &mut |cur, total| {
            let _ = progress_callback.call2(
                &this, &JsValue::from(cur as u32), &JsValue::from(total as u32),
            );
        },
    );

    match result {
        Ok(chi) => chi,
        Err(e) => {
            console_log!("run_dipole_inversion_wasm error: {}", e);
            vec![0.0; n_total]
        }
    }
}

/// Apply QSM referencing (mean subtraction or none).
#[wasm_bindgen]
pub fn apply_reference_wasm(chi: &[f64], mask: &[u8], method: &str) -> Vec<f64> {
    let ref_method = match method {
        "none" => qsm_core::pipeline::QsmReference::None,
        _ => qsm_core::pipeline::QsmReference::Mean,
    };
    qsm_core::pipeline::apply_reference(chi, mask, ref_method)
}

/// Scale phase data to [-pi, pi] range in-place and return the result.
#[wasm_bindgen]
pub fn scale_phase_to_pi_wasm(phase: &[f64]) -> Vec<f64> {
    let mut data = phase.to_vec();
    qsm_core::pipeline::scale_phase_to_pi(&mut data);
    data
}

/// Convert Hz field to ppm given field strength.
#[wasm_bindgen]
pub fn hz_to_ppm_wasm(field_hz: &[f64], field_strength: f64) -> Vec<f64> {
    qsm_core::pipeline::hz_to_ppm(field_hz, field_strength)
}

/// Convert rad/s field to ppm given field strength.
#[wasm_bindgen]
pub fn rads_to_ppm_wasm(field_rads: &[f64], field_strength: f64) -> Vec<f64> {
    qsm_core::pipeline::rads_to_ppm(field_rads, field_strength)
}
