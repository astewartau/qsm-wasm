/**
 * QSM Processing Web Worker
 *
 * Runs Pyodide and the QSM pipeline in a separate thread
 * to keep the main UI responsive.
 */

let pyodide = null;

// Post progress updates to main thread
function postProgress(value, text) {
  self.postMessage({ type: 'progress', value, text });
}

// Post log messages to main thread
function postLog(message) {
  self.postMessage({ type: 'log', message });
}

// Post results to main thread
function postResult(stage, data) {
  self.postMessage({ type: 'result', stage, data });
}

// Post error to main thread
function postError(message) {
  self.postMessage({ type: 'error', message });
}

// Post completion to main thread
function postComplete(results) {
  self.postMessage({ type: 'complete', results });
}

async function initializePyodide() {
  postLog("Initializing Pyodide...");
  postProgress(0.05, 'Loading Pyodide...');

  importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.js');
  pyodide = await loadPyodide();

  postLog("Installing Python packages...");
  postProgress(0.07, 'Installing packages...');
  await pyodide.loadPackage(["numpy", "scipy", "micropip"]);

  postLog("Installing nibabel...");
  await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("nibabel")
  `);

  postLog("Pyodide ready");
  return pyodide;
}

async function loadRomeoCode(romeoCode) {
  postLog("Loading ROMEO algorithm...");
  await pyodide.runPython(romeoCode);
}

async function runPipeline(data) {
  const { magnitudeBuffers, phaseBuffers, echoTimes, magField, unwrapMode, maskThreshold } = data;
  const thresholdFraction = (maskThreshold || 15) / 100;  // Default to 15% if not provided

  try {
    postProgress(0.1, 'Loading data...');
    postLog("Loading multi-echo data...");

    // Transfer data to Python
    for (let i = 0; i < echoTimes.length; i++) {
      pyodide.globals.set(`mag_data_${i}`, new Uint8Array(magnitudeBuffers[i]));
      pyodide.globals.set(`phase_data_${i}`, new Uint8Array(phaseBuffers[i]));
    }

    pyodide.globals.set("echo_times", new Float64Array(echoTimes));
    pyodide.globals.set("num_echoes", echoTimes.length);

    // Load data in Python
    await pyodide.runPython(`
import numpy as np
import nibabel as nib
from io import BytesIO

# Convert JsProxy objects to Python objects
echo_times_py = echo_times.to_py()

print(f"Loading {num_echoes} echoes...")
print(f"Echo times: {list(echo_times_py)} ms")

# Load magnitude and phase data
magnitude_4d = []
phase_4d = []

for i in range(num_echoes):
    # Load magnitude
    mag_bytes = globals()[f'mag_data_{i}'].to_py()
    mag_fh = nib.FileHolder(BytesIO(mag_bytes))
    mag_img = nib.Nifti1Image.from_file_map({'image': mag_fh, 'header': mag_fh})
    mag_data = mag_img.get_fdata()
    magnitude_4d.append(mag_data)

    # Load phase
    phase_bytes = globals()[f'phase_data_{i}'].to_py()
    phase_fh = nib.FileHolder(BytesIO(phase_bytes))
    phase_img = nib.Nifti1Image.from_file_map({'image': phase_fh, 'header': phase_fh})
    phase_data = phase_img.get_fdata()

    # Wrap phase to [-π, π] using complex exponential
    # This handles both arbitrary scaling and extended phase ranges
    if np.max(np.abs(phase_data)) > np.pi * 1.1:
        if np.max(np.abs(phase_data)) > 10:
            # Data is in arbitrary units (e.g., 0-4095), scale to [-π, π]
            phase_data = (phase_data - np.min(phase_data)) / (np.max(phase_data) - np.min(phase_data)) * 2 * np.pi - np.pi
        # Now wrap to [-π, π]
        phase_data = np.angle(np.exp(1j * phase_data))

    phase_4d.append(phase_data)

# Stack into 4D arrays
magnitude_4d = np.stack(magnitude_4d, axis=3)
phase_4d = np.stack(phase_4d, axis=3)

print(f"Data shape: {magnitude_4d.shape}")
print(f"Phase range: [{np.min(phase_4d):.3f}, {np.max(phase_4d):.3f}]")
print(f"Magnitude range: [{np.min(magnitude_4d):.1f}, {np.max(magnitude_4d):.1f}]")

# Store header info from first echo
header_info = mag_img.header
affine_matrix = mag_img.affine
`);

    postProgress(0.1, 'Unwrapping phase...');
    postLog("Running ROMEO phase unwrapping...");

    // Create progress callback for Python to call
    // ROMEO takes ~75% of total time, maps to 1-76% of progress bar
    const unwrapProgressCallback = (stage, progress) => {
      const mappedProgress = 0.01 + (progress / 100) * 0.75;
      postProgress(mappedProgress, `Unwrapping: ${progress}%`);
    };
    pyodide.globals.set('js_progress_callback', unwrapProgressCallback);

    const individual = unwrapMode === 'individual';
    await pyodide.runPython(`
print("Starting ROMEO unwrapping...")
print(f"Mode: {'Individual' if ${individual ? 'True' : 'False'} else 'Temporal'}")

# Set up progress callback
set_progress_callback(js_progress_callback)

results = romeo_multi_echo_unwrap(
    phase_4d, magnitude_4d, echo_times_py,
    individual=${individual ? 'True' : 'False'},
    B0_calculation=True,
    weighting='phase_snr'
)

report_progress("complete", 100)

unwrapped_phase = results['unwrapped']
B0_fieldmap = results['B0']
processing_mask = results['mask']

print(f"ROMEO unwrapping completed!")
print(f"B0 range: [{np.min(B0_fieldmap):.1f}, {np.max(B0_fieldmap):.1f}] Hz")

# Use first echo magnitude and B0 fieldmap for subsequent processing
magnitude_combined = magnitude_4d[:, :, :, 0]  # First echo magnitude
fieldmap = B0_fieldmap  # B0 fieldmap from ROMEO

print(f"Using first echo magnitude: {magnitude_combined.shape}")
print(f"Using B0 fieldmap: {fieldmap.shape}")
`);

    postProgress(0.76, 'Removing background...');
    postLog("Removing background field...");

    // Create progress callback for V-SHARP
    // V-SHARP takes ~8% of total time, maps to 76-84% of progress bar
    const vsharpProgressCallback = (current, total) => {
      const progress = 0.76 + (current / total) * 0.08;
      postProgress(progress, `V-SHARP: radius ${current}/${total}`);
    };
    pyodide.globals.set('js_vsharp_progress', vsharpProgressCallback);

    await pyodide.runPython(`
# V-SHARP background removal (Variable-kernel SHARP)
import numpy as np

print("Running V-SHARP background removal...")

def create_smv_kernel_kspace(shape, voxel_size, radius):
    """Create spherical mean value kernel in k-space (VECTORIZED)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    r2 = radius * radius

    # Create coordinate grids (vectorized)
    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    # Wrap coordinates around center
    x = np.where(i <= nx//2, i, i - nx) * dx
    y = np.where(j <= ny//2, j, j - ny) * dy
    z = np.where(k <= nz//2, k, k - nz) * dz

    # Create 3D grids
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sphere mask (vectorized)
    sphere = (X*X + Y*Y + Z*Z <= r2).astype(np.float64)

    # Normalize
    sphere_sum = np.sum(sphere)
    if sphere_sum > 0:
        sphere /= sphere_sum

    # FFT to get k-space kernel
    S = np.real(np.fft.fftn(sphere))
    return S

def vsharp_background_removal(fieldmap, mask, voxel_size=(1.0, 1.0, 1.0),
                               radii=None, threshold=0.05):
    """
    V-SHARP background field removal

    Uses multiple kernel sizes from large to small to preserve more brain tissue.
    Based on: Wu B, et al. MRM 2012;67(1):137-47
    """
    if radii is None:
        # Default radii: 18mm down to 2mm in steps of 2mm
        min_vox = min(voxel_size)
        max_vox = max(voxel_size)
        radii = list(np.arange(18*min_vox, 2*max_vox - 0.001, -2*max_vox))
        if len(radii) == 0:
            radii = [6.0, 4.0, 2.0]

    # Sort radii from largest to smallest
    radii = sorted(radii, reverse=True)
    print(f"V-SHARP radii (mm): {[f'{r:.1f}' for r in radii]}")

    shape = fieldmap.shape

    # FFT of field
    F_hat = np.fft.fftn(fieldmap)

    # FFT of mask (as float)
    M_hat = np.fft.fftn(mask.astype(np.float64))

    # Output arrays
    local_field = np.zeros_like(fieldmap)
    final_mask = np.zeros(shape, dtype=bool)
    prev_mask = np.zeros(shape, dtype=bool)

    # Threshold for mask erosion (voxels with SMV(mask) > delta are inside)
    delta = 1.0 - np.sqrt(np.finfo(np.float64).eps)

    # Store inverse high-pass filter from largest kernel for final deconvolution
    iS_largest = None

    n_radii = len(radii)
    for i, radius in enumerate(radii):
        # Report progress
        try:
            js_vsharp_progress(i + 1, n_radii)
        except:
            pass
        print(f"  Processing radius {i+1}/{n_radii}: {radius:.1f} mm")
        # Create SMV kernel in k-space
        S = create_smv_kernel_kspace(shape, voxel_size, radius)

        # High-pass filter: 1 - S
        HP = 1.0 - S

        # Erode mask: IFFT(S * M_hat) and threshold
        eroded = np.real(np.fft.ifftn(S * M_hat))
        current_mask = eroded > delta

        # Store inverse of HP filter from largest kernel (for final deconvolution)
        if i == 0:
            # Threshold to avoid division by small values
            iS_largest = np.where(np.abs(HP) < threshold, 0.0, 1.0 / HP)

        # Apply high-pass filter to field
        F_hp = HP * F_hat

        # IFFT to get high-pass filtered field
        hp_field = np.real(np.fft.ifftn(F_hp))

        # Fill in voxels that are in current eroded mask but not in previous (smaller) mask
        new_voxels = current_mask & ~prev_mask
        local_field[new_voxels] = hp_field[new_voxels]

        # Update masks
        prev_mask = current_mask.copy()
        final_mask = current_mask

    # Final deconvolution with largest kernel's inverse HP filter
    F_local = np.fft.fftn(local_field)
    F_local *= iS_largest
    local_field = np.real(np.fft.ifftn(F_local))

    # Apply final eroded mask
    local_field[~final_mask] = 0

    return local_field, final_mask

# Create processing mask using user-defined threshold
mask_threshold_fraction = ${thresholdFraction}
print(f"Using mask threshold: {mask_threshold_fraction * 100:.0f}% of max magnitude")

# Always use user-defined threshold (override ROMEO mask)
processing_mask = magnitude_combined > mask_threshold_fraction * np.max(magnitude_combined)

print(f"Processing mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")

# Get voxel size from header if available, otherwise assume 1mm isotropic
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
    print(f"Voxel size: {voxel_size} mm")
except:
    voxel_size = (1.0, 1.0, 1.0)
    print(f"Using default voxel size: {voxel_size} mm")

# Run V-SHARP background removal
local_fieldmap, eroded_mask = vsharp_background_removal(
    fieldmap, processing_mask, voxel_size=voxel_size
)

# Update processing mask to eroded version
processing_mask = eroded_mask

print(f"Eroded mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")
print(f"Local field range: [{np.min(local_fieldmap[processing_mask]):.1f}, {np.max(local_fieldmap[processing_mask]):.1f}] Hz")
print("V-SHARP background removal completed!")
`);

    postProgress(0.8, 'Dipole inversion...');
    postLog("Running QSM dipole inversion...");

    // Create progress callback for QSM
    // QSM takes ~16% of total time, maps to 84-100% of progress bar
    const qsmProgressCallback = (iteration, maxiter) => {
      const progress = 0.84 + (iteration / maxiter) * 0.16;
      postProgress(progress, `QSM: iteration ${iteration}/${maxiter}`);
    };
    pyodide.globals.set('js_qsm_progress', qsmProgressCallback);

    await pyodide.runPython(`
# RTS (Rapid Two-Step) dipole inversion for QSM
# Based on: Kames C, et al. Neuroimage 2018;167:276-83
print("Running RTS QSM dipole inversion...")

def create_dipole_kernel(shape, voxel_size, bdir=(0, 0, 1)):
    """Create dipole kernel in k-space"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    # Create k-space grid (centered at DC)
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Normalize B field direction
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)

    # k dot B
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]

    # |k|^2
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12  # Avoid division by zero at DC

    # Dipole kernel: D = 1/3 - (k·B)²/|k|²
    D = 1/3 - (k_dot_b**2) / k2

    # Set DC to zero
    D[0, 0, 0] = 0

    return D

def create_laplacian_kernel(shape, voxel_size):
    """Create negative Laplacian kernel in k-space for gradient regularization"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # -Laplacian in k-space: 4π²(kx² + ky² + kz²)
    # Using discrete Laplacian: 2*(1-cos(2πk*h))/h² for each dimension
    Lx = 2 * (1 - np.cos(2 * np.pi * KX * dx)) / (dx * dx)
    Ly = 2 * (1 - np.cos(2 * np.pi * KY * dy)) / (dy * dy)
    Lz = 2 * (1 - np.cos(2 * np.pi * KZ * dz)) / (dz * dz)

    return Lx + Ly + Lz

def rts_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
            delta=0.15, mu=1e5, rho=10.0, tol=1e-2, maxit=20):
    """
    RTS (Rapid Two-Step) QSM dipole inversion

    Step 1: Direct inversion for well-conditioned k-space (|D| > delta)
    Step 2: ADMM with TV regularization for ill-conditioned regions

    Based on: Kames C, et al. Neuroimage 2018;167:276-83
    """
    shape = local_field.shape
    nx, ny, nz = shape

    # Create kernels
    D = create_dipole_kernel(shape, voxel_size, bdir)
    L = create_laplacian_kernel(shape, voxel_size)

    # Mask of well-conditioned frequencies
    well_conditioned = np.abs(D) > delta
    M = np.where(well_conditioned, mu, 0.0)

    # Apply mask to field
    f = local_field * mask

    # FFT of masked field
    F = np.fft.fftn(f)

    ######################################################################
    # Step 1: Well-conditioned k-space - direct division with thresholding
    ######################################################################
    print("Step 1: Direct inversion for well-conditioned k-space...")

    # Simple LSMR-like iteration (conjugate of D * F / |D|²)
    # For well-conditioned: x = D* F / |D|²
    D_sq = D * D
    D_sq_safe = np.where(D_sq > delta**2, D_sq, delta**2)

    X = D * F / D_sq_safe  # Initial estimate
    x = np.real(np.fft.ifftn(X))
    x = x * mask

    ######################################################################
    # Step 2: Ill-conditioned k-space - ADMM with TV regularization
    ######################################################################
    print("Step 2: ADMM with TV for ill-conditioned k-space...")

    # Pre-compute: iA = rho / (M + rho * L)
    denominator = M + rho * L
    denominator[denominator == 0] = 1e-12
    iA = rho / denominator

    # Pre-compute constant term: F_const = M * X / (M + rho * L)
    X = np.fft.fftn(x)
    F_const = M * X / denominator

    # Initialize dual variables (gradients)
    px = np.zeros(shape)
    py = np.zeros(shape)
    pz = np.zeros(shape)

    # Gradient operators using finite differences
    def gradient(x):
        gx = np.roll(x, -1, axis=0) - x
        gy = np.roll(x, -1, axis=1) - x
        gz = np.roll(x, -1, axis=2) - x
        return gx, gy, gz

    def divergence(px, py, pz):
        # Negative adjoint of gradient
        dx = px - np.roll(px, 1, axis=0)
        dy = py - np.roll(py, 1, axis=1)
        dz = pz - np.roll(pz, 1, axis=2)
        return dx + dy + dz

    def shrink(x, thresh):
        # Soft thresholding
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

    inv_rho = 1.0 / rho
    x_prev = x.copy()

    for iteration in range(maxit):
        # Report progress
        try:
            js_qsm_progress(iteration + 1, maxit)
        except:
            pass

        # Compute gradients
        gx, gy, gz = gradient(x)

        # y-subproblem: soft thresholding
        yx = shrink(gx + px, inv_rho)
        yy = shrink(gy + py, inv_rho)
        yz = shrink(gz + pz, inv_rho)

        # Update dual variables
        px = px + gx - yx
        py = py + gy - yy
        pz = pz + gz - yz

        # x-subproblem: solve in k-space
        # x = iFFT( iA * FFT(div(y - p)) + F_const )
        vx = yx - px
        vy = yy - py
        vz = yz - pz
        div_v = divergence(vx, vy, vz)

        X = iA * np.fft.fftn(div_v) + F_const
        x = np.real(np.fft.ifftn(X))

        # Check convergence
        diff = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-12)
        if diff < tol:
            print(f"  Converged at iteration {iteration + 1}, diff = {diff:.2e}")
            break
        x_prev = x.copy()

        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}/{maxit}, diff = {diff:.2e}")

    # Apply mask
    x = x * mask

    return x

# Get voxel size
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
except:
    voxel_size = (1.0, 1.0, 1.0)

print(f"Voxel size: {voxel_size} mm")
print(f"B0 field strength: ${magField} T")

# Run RTS QSM
qsm_result = rts_qsm(
    local_fieldmap,
    processing_mask,
    voxel_size,
    delta=0.15,
    mu=1e5,
    rho=10.0,
    maxit=20
)

# Scale to ppm (susceptibility is dimensionless, but often expressed in ppm)
# The local field is in Hz, need to convert via: chi = f / (gamma * B0)
# gamma = 42.576 MHz/T, so gamma * B0 (in Hz) for B0 in Tesla
gamma_B0 = 42.576e6 * ${magField}  # Hz
# chi (ppm) = f (Hz) / (gamma * B0) * 1e6
# But our inversion already accounts for the dipole relationship
# The result is already in the correct susceptibility units relative to the input field

print(f"QSM result range: [{np.min(qsm_result[processing_mask]):.4f}, {np.max(qsm_result[processing_mask]):.4f}]")
print("RTS QSM dipole inversion completed!")

# Final cleanup
qsm_result[~processing_mask] = 0
`);

    postProgress(1.0, 'Complete');
    postLog("Pipeline completed successfully!");
    postComplete({ success: true });

  } catch (error) {
    postError(error.message);
    throw error;
  }
}

async function getStageData(stage) {
  let dataName, description;

  switch (stage) {
    case 'magnitude':
      dataName = 'magnitude_combined';
      description = 'Magnitude (First Echo)';
      break;
    case 'phase':
      dataName = 'phase_4d[:,:,:,0]';
      description = 'Phase (First Echo)';
      break;
    case 'mask':
      dataName = 'processing_mask.astype(np.float32)';
      description = 'Processing Mask';
      break;
    case 'B0':
      dataName = 'B0_fieldmap';
      description = 'B0 Field Map';
      break;
    case 'bgRemoved':
      dataName = 'local_fieldmap';
      description = 'Local Field Map';
      break;
    case 'final':
      dataName = 'qsm_result';
      description = 'QSM Result';
      break;
    default:
      throw new Error(`Unknown stage: ${stage}`);
  }

  await pyodide.runPython(`
import nibabel as nib

# Get the data
display_data = ${dataName}
print(f"Exporting {${JSON.stringify(description)}}: shape {display_data.shape}")

# Create NIfTI file
nii_img = nib.Nifti1Image(display_data, affine_matrix, header_info)

# Save to bytes
import tempfile
import os
temp_path = '/tmp/temp_output.nii'
nii_img.to_filename(temp_path)

# Read the file as bytes
with open(temp_path, 'rb') as f:
    output_bytes = f.read()

# Clean up
os.remove(temp_path)
`);

  const outputBytes = pyodide.globals.get('output_bytes').toJs();
  return { stage, data: outputBytes, description };
}

// Handle messages from main thread
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init':
        await initializePyodide();
        await loadRomeoCode(data.romeoCode);
        self.postMessage({ type: 'initialized' });
        break;

      case 'run':
        await runPipeline(data);
        break;

      case 'getStage':
        const result = await getStageData(data.stage);
        self.postMessage({ type: 'stageData', ...result });
        break;

      default:
        postError(`Unknown message type: ${type}`);
    }
  } catch (error) {
    postError(error.message);
    console.error(error);
  }
};
