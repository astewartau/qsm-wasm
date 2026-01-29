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
  const { magnitudeBuffers, phaseBuffers, echoTimes, magField, unwrapMode, maskThreshold, customMaskBuffer, pipelineSettings } = data;
  const thresholdFraction = (maskThreshold || 15) / 100;  // Default to 15% if not provided
  const hasCustomMask = customMaskBuffer !== null && customMaskBuffer !== undefined;

  // Extract pipeline settings with defaults
  const unwrapMethod = pipelineSettings?.unwrapMethod || 'romeo';
  const romeoSettings = pipelineSettings?.romeo || { weighting: 'phase_snr' };
  const backgroundMethod = pipelineSettings?.backgroundRemoval || 'vsharp';
  const vsharpSettings = pipelineSettings?.vsharp || { maxRadius: 18, minRadius: 2, threshold: 0.05 };
  const smvSettings = pipelineSettings?.smv || { radius: 5 };
  const dipoleMethod = pipelineSettings?.dipoleInversion || 'rts';
  const rtsSettings = pipelineSettings?.rts || { delta: 0.15, mu: 100000, rho: 10, maxIter: 20 };
  const mediSettings = pipelineSettings?.medi || { lambda: 1000, maxIter: 10, cgMaxIter: 100, cgTol: 0.01, edgePercent: 0.9, merit: false };
  const tkdSettings = pipelineSettings?.tkd || { threshold: 0.15 };
  const tikhonovSettings = pipelineSettings?.tikhonov || { lambda: 0.01, reg: 'identity' };
  const tvSettings = pipelineSettings?.tv || { lambda: 0.001, maxIter: 50, tol: 0.001 };

  try {
    postProgress(0.1, 'Loading data...');
    postLog("Loading multi-echo data...");

    // Transfer data to Python
    for (let i = 0; i < echoTimes.length; i++) {
      pyodide.globals.set(`mag_data_${i}`, new Uint8Array(magnitudeBuffers[i]));
      pyodide.globals.set(`phase_data_${i}`, new Uint8Array(phaseBuffers[i]));
    }

    // Transfer custom mask if provided
    if (hasCustomMask) {
      pyodide.globals.set('custom_mask_data', new Uint8Array(customMaskBuffer));
      postLog("Using custom edited mask");
    }
    pyodide.globals.set('has_custom_mask', hasCustomMask);

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

    // Choose unwrapping method
    const useRomeo = unwrapMethod === 'romeo';
    pyodide.globals.set('use_romeo_unwrap', useRomeo);

    if (useRomeo) {
      postLog("Running ROMEO phase unwrapping...");

      // Create progress callback for Python to call
      // ROMEO takes ~75% of total time, maps to 1-76% of progress bar
      const unwrapProgressCallback = (stage, progress) => {
        const mappedProgress = 0.01 + (progress / 100) * 0.75;
        postProgress(mappedProgress, `Unwrapping: ${progress}%`);
      };
      pyodide.globals.set('js_progress_callback', unwrapProgressCallback);

      const individual = unwrapMode === 'individual';
      const romeoWeighting = romeoSettings.weighting;
      pyodide.globals.set('romeo_weighting', romeoWeighting);
      await pyodide.runPython(`
print("Starting ROMEO unwrapping...")
print(f"Mode: {'Individual' if ${individual ? 'True' : 'False'} else 'Temporal'}")
print(f"Weighting: {romeo_weighting}")

# Set up progress callback
set_progress_callback(js_progress_callback)

results = romeo_multi_echo_unwrap(
    phase_4d, magnitude_4d, echo_times_py,
    individual=${individual ? 'True' : 'False'},
    B0_calculation=True,
    weighting=romeo_weighting
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
    } else {
      // Laplacian unwrapping
      postLog("Running Laplacian phase unwrapping...");

      await pyodide.runPython(`
print("Starting Laplacian unwrapping...")

def laplacian_unwrap(wrapped_phase, voxel_size=(1.0, 1.0, 1.0)):
    """
    Laplacian phase unwrapping using Fourier-domain Poisson solver.

    The method exploits that the Laplacian of the true phase equals
    the Laplacian of the wrapped phase (computed via complex derivatives).

    Based on: Schofield & Zhu, Optics Letters 2003
    """
    shape = wrapped_phase.shape

    # Compute cosine and sine of wrapped phase
    cos_phi = np.cos(wrapped_phase)
    sin_phi = np.sin(wrapped_phase)

    # Compute Laplacian using finite differences (7-point stencil)
    def laplacian_3d(f, voxel_size):
        dx, dy, dz = voxel_size
        lap = np.zeros_like(f)
        # x direction
        lap += (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dx*dx)
        # y direction
        lap += (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (dy*dy)
        # z direction
        lap += (np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / (dz*dz)
        return lap

    # Compute Laplacian of cosine and sine
    lap_cos = laplacian_3d(cos_phi, voxel_size)
    lap_sin = laplacian_3d(sin_phi, voxel_size)

    # Compute Laplacian of unwrapped phase
    # Derivation: sin(φ)∇²cos(φ) - cos(φ)∇²sin(φ) = -∇²φ
    # So: ∇²φ = cos(φ)∇²sin(φ) - sin(φ)∇²cos(φ)
    lap_phi = cos_phi * lap_sin - sin_phi * lap_cos

    # Create Laplacian kernel in Fourier space
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Discrete Laplacian in frequency domain (matches finite difference stencil)
    # Note: eigenvalues are negative (or zero at DC)
    L = (2 * (np.cos(2 * np.pi * KX * dx) - 1) / (dx * dx) +
         2 * (np.cos(2 * np.pi * KY * dy) - 1) / (dy * dy) +
         2 * (np.cos(2 * np.pi * KZ * dz) - 1) / (dz * dz))

    # Regularize to avoid division by very small values
    # Add small positive value to make all entries non-zero
    eps = 1e-6
    L_reg = np.where(np.abs(L) < eps, -eps, L)

    # Solve Poisson equation: ∇²φ = lap_phi  =>  φ = F⁻¹[F(lap_phi) / L]
    F_lap_phi = np.fft.fftn(lap_phi)
    F_phi = F_lap_phi / L_reg
    F_phi[0, 0, 0] = 0  # Set DC to zero (removes global offset)

    unwrapped = np.real(np.fft.ifftn(F_phi))

    print(f"  Debug - wrapped range: [{np.min(wrapped_phase):.4f}, {np.max(wrapped_phase):.4f}]")
    print(f"  Debug - lap_phi range: [{np.min(lap_phi):.4f}, {np.max(lap_phi):.4f}]")
    print(f"  Debug - L range: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"  Debug - unwrapped range: [{np.min(unwrapped):.4f}, {np.max(unwrapped):.4f}]")

    return unwrapped

def compute_b0_from_unwrapped(unwrapped_4d, echo_times_ms, mask):
    """Compute B0 field map from unwrapped phase.

    For single echo: B0 = phase / (2π * TE)
    For multi-echo: weighted linear fit of phase vs TE
    """
    n_echoes = unwrapped_4d.shape[3]
    echo_times_s = np.array(echo_times_ms) / 1000.0  # Convert to seconds

    if n_echoes == 1:
        # Single echo: direct conversion
        # phase (rad) = 2π * B0 (Hz) * TE (s)
        # B0 (Hz) = phase / (2π * TE)
        TE = echo_times_s[0]
        B0_Hz = unwrapped_4d[:,:,:,0] / (2 * np.pi * TE)
        print(f"  Single echo B0 computation: TE={TE*1000:.1f}ms")
    else:
        # Multi-echo: weighted linear fit
        # phase = B0 * TE + phase0
        weights = np.arange(1, n_echoes + 1, dtype=np.float64)
        weights = weights / np.sum(weights)

        te_mean = np.sum(weights * echo_times_s)

        numerator = np.zeros(unwrapped_4d.shape[:3])
        denominator = 0.0

        for i in range(n_echoes):
            te_diff = echo_times_s[i] - te_mean
            numerator += weights[i] * te_diff * unwrapped_4d[:,:,:,i]
            denominator += weights[i] * te_diff * te_diff

        B0_rad_per_s = numerator / (denominator + 1e-10)
        B0_Hz = B0_rad_per_s / (2 * np.pi)
        print(f"  Multi-echo B0 computation: {n_echoes} echoes")

    B0_Hz = B0_Hz * mask
    return B0_Hz

# Get voxel size
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
except:
    voxel_size = (1.0, 1.0, 1.0)

print(f"Voxel size: {voxel_size}")

# Unwrap each echo
n_echoes = phase_4d.shape[3]
unwrapped_4d = np.zeros_like(phase_4d)

for i in range(n_echoes):
    print(f"  Unwrapping echo {i+1}/{n_echoes}...")
    unwrapped_4d[:,:,:,i] = laplacian_unwrap(phase_4d[:,:,:,i], voxel_size)

print("Laplacian unwrapping completed!")

# Create mask from magnitude (threshold-based)
magnitude_combined = magnitude_4d[:, :, :, 0]  # First echo
mag_threshold = np.max(magnitude_combined) * ${thresholdFraction}
processing_mask = magnitude_combined > mag_threshold

# Compute B0 fieldmap from unwrapped phase
print("Computing B0 field map...")
B0_fieldmap = compute_b0_from_unwrapped(unwrapped_4d, echo_times_py, processing_mask)
fieldmap = B0_fieldmap

print(f"B0 range: [{np.min(B0_fieldmap[processing_mask]):.1f}, {np.max(B0_fieldmap[processing_mask]):.1f}] Hz")
print(f"Using first echo magnitude: {magnitude_combined.shape}")
`);
    }

    postProgress(0.76, 'Removing background...');
    postLog(`Removing background field using ${backgroundMethod.toUpperCase()}...`);

    // Create progress callback for background removal
    const bgProgressCallback = (current, total) => {
      const progress = 0.76 + (current / total) * 0.08;
      postProgress(progress, `${backgroundMethod.toUpperCase()}: ${current}/${total}`);
    };
    pyodide.globals.set('js_bg_progress', bgProgressCallback);
    pyodide.globals.set('background_method', backgroundMethod);

    await pyodide.runPython(`
# Background removal - V-SHARP or SMV
import numpy as np

def create_smv_kernel_kspace(shape, voxel_size, radius):
    """Create spherical mean value kernel in k-space (VECTORIZED)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    r2 = radius * radius

    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    x = np.where(i <= nx//2, i, i - nx) * dx
    y = np.where(j <= ny//2, j, j - ny) * dy
    z = np.where(k <= nz//2, k, k - nz) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    sphere = (X*X + Y*Y + Z*Z <= r2).astype(np.float64)

    sphere_sum = np.sum(sphere)
    if sphere_sum > 0:
        sphere /= sphere_sum

    S = np.real(np.fft.fftn(sphere))
    return S

def vsharp_background_removal(fieldmap, mask, voxel_size=(1.0, 1.0, 1.0),
                               radii=None, threshold=0.05):
    """V-SHARP background field removal with variable radii"""
    if radii is None:
        min_vox = min(voxel_size)
        max_vox = max(voxel_size)
        radii = list(np.arange(18*min_vox, 2*max_vox - 0.001, -2*max_vox))
        if len(radii) == 0:
            radii = [6.0, 4.0, 2.0]

    radii = sorted(radii, reverse=True)
    print(f"V-SHARP radii (mm): {[f'{r:.1f}' for r in radii]}")

    shape = fieldmap.shape
    F_hat = np.fft.fftn(fieldmap)
    M_hat = np.fft.fftn(mask.astype(np.float64))

    local_field = np.zeros_like(fieldmap)
    final_mask = np.zeros(shape, dtype=bool)
    prev_mask = np.zeros(shape, dtype=bool)
    delta = 1.0 - np.sqrt(np.finfo(np.float64).eps)
    iS_largest = None

    n_radii = len(radii)
    for i, radius in enumerate(radii):
        try:
            js_bg_progress(i + 1, n_radii)
        except:
            pass
        print(f"  Processing radius {i+1}/{n_radii}: {radius:.1f} mm")
        S = create_smv_kernel_kspace(shape, voxel_size, radius)
        HP = 1.0 - S
        eroded = np.real(np.fft.ifftn(S * M_hat))
        current_mask = eroded > delta

        if i == 0:
            iS_largest = np.where(np.abs(HP) < threshold, 0.0, 1.0 / HP)

        F_hp = HP * F_hat
        hp_field = np.real(np.fft.ifftn(F_hp))
        new_voxels = current_mask & ~prev_mask
        local_field[new_voxels] = hp_field[new_voxels]
        prev_mask = current_mask.copy()
        final_mask = current_mask

    F_local = np.fft.fftn(local_field)
    F_local *= iS_largest
    local_field = np.real(np.fft.ifftn(F_local))
    local_field[~final_mask] = 0

    return local_field, final_mask

def smv_background_removal(fieldmap, mask, voxel_size, radius=5.0):
    """SMV background field removal with single radius"""
    print(f"SMV background removal with radius={radius:.1f}mm")
    shape = fieldmap.shape

    try:
        js_bg_progress(1, 3)
    except:
        pass

    S = create_smv_kernel_kspace(shape, voxel_size, radius)

    try:
        js_bg_progress(2, 3)
    except:
        pass

    HP = 1.0 - S
    F = np.fft.fftn(fieldmap)
    local_field = np.real(np.fft.ifftn(HP * F))

    M = np.fft.fftn(mask.astype(np.float64))
    eroded = np.real(np.fft.ifftn(S * M))
    eroded_mask = eroded > 0.999

    local_field = local_field * eroded_mask

    try:
        js_bg_progress(3, 3)
    except:
        pass

    return local_field, eroded_mask

# Create processing mask - use custom mask if provided, otherwise threshold
if has_custom_mask:
    print("Loading custom edited mask...")
    mask_bytes = custom_mask_data.to_py()
    mask_fh = nib.FileHolder(BytesIO(mask_bytes))
    mask_img = nib.Nifti1Image.from_file_map({'image': mask_fh, 'header': mask_fh})
    mask_data = mask_img.get_fdata()
    processing_mask = mask_data > 0.5
    print(f"Custom mask loaded: {processing_mask.shape}")
else:
    mask_threshold_fraction = ${thresholdFraction}
    print(f"Using mask threshold: {mask_threshold_fraction * 100:.0f}% of max magnitude")
    processing_mask = magnitude_combined > mask_threshold_fraction * np.max(magnitude_combined)

print(f"Processing mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")

# Get voxel size from header
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
    print(f"Voxel size: {voxel_size} mm")
except:
    voxel_size = (1.0, 1.0, 1.0)
    print(f"Using default voxel size: {voxel_size} mm")

# Run background removal based on selected method
bg_method = background_method
if bg_method == 'vsharp':
    print("Running V-SHARP background removal...")
    vsharp_max_radius = ${vsharpSettings.maxRadius}
    vsharp_min_radius = ${vsharpSettings.minRadius}
    vsharp_threshold = ${vsharpSettings.threshold}
    print(f"V-SHARP settings: max_radius={vsharp_max_radius}mm, min_radius={vsharp_min_radius}mm, threshold={vsharp_threshold}")
    radii = list(np.arange(vsharp_max_radius, vsharp_min_radius - 0.001, -2.0))
    local_fieldmap, eroded_mask = vsharp_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        radii=radii, threshold=vsharp_threshold
    )
else:  # smv
    print("Running SMV background removal...")
    smv_radius = ${smvSettings.radius}
    print(f"SMV settings: radius={smv_radius}mm")
    local_fieldmap, eroded_mask = smv_background_removal(
        fieldmap, processing_mask, voxel_size=voxel_size,
        radius=smv_radius
    )

# Update processing mask to eroded version
processing_mask = eroded_mask

print(f"Eroded mask coverage: {np.sum(processing_mask)}/{processing_mask.size} voxels ({100*np.sum(processing_mask)/processing_mask.size:.1f}%)")
print(f"Local field range: [{np.min(local_fieldmap[processing_mask]):.1f}, {np.max(local_fieldmap[processing_mask]):.1f}] Hz")
print("Background removal completed!")
`);

    postProgress(0.8, 'Dipole inversion...');
    postLog(`Running ${dipoleMethod.toUpperCase()} dipole inversion...`);

    // Create progress callback for QSM
    const qsmProgressCallback = (iteration, maxiter) => {
      const progress = 0.84 + (iteration / maxiter) * 0.16;
      postProgress(progress, `${dipoleMethod.toUpperCase()}: iteration ${iteration}/${maxiter}`);
    };
    pyodide.globals.set('js_qsm_progress', qsmProgressCallback);
    pyodide.globals.set('dipole_method', dipoleMethod);

    await pyodide.runPython(`
# Dipole inversion - RTS or MEDI

def create_dipole_kernel(shape, voxel_size, bdir=(0, 0, 1)):
    """Create dipole kernel in k-space (full FFT version)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12
    D = 1/3 - (k_dot_b**2) / k2
    D[0, 0, 0] = 0
    return D

def create_dipole_kernel_rfft(shape, voxel_size, bdir=(0, 0, 1)):
    """Create dipole kernel for rfft (half-spectrum, ~2x faster)"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    # Full frequencies for x and y, half for z (rfft)
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.rfftfreq(nz, dz)  # Only positive frequencies
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12
    D = 1/3 - (k_dot_b**2) / k2
    D[0, 0, 0] = 0
    return D

def create_laplacian_kernel(shape, voxel_size):
    """Create negative Laplacian kernel in k-space"""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    Lx = 2 * (1 - np.cos(2 * np.pi * KX * dx)) / (dx * dx)
    Ly = 2 * (1 - np.cos(2 * np.pi * KY * dy)) / (dy * dy)
    Lz = 2 * (1 - np.cos(2 * np.pi * KZ * dz)) / (dz * dz)
    return Lx + Ly + Lz

# --- Helper functions for TV-ADMM (from QSM.jl fd.jl) ---
def gradient_periodic(x, voxel_size):
    """Forward gradient with periodic boundaries"""
    dx, dy, dz = voxel_size
    gx = (np.roll(x, -1, axis=0) - x) / dx
    gy = (np.roll(x, -1, axis=1) - x) / dy
    gz = (np.roll(x, -1, axis=2) - x) / dz
    return gx, gy, gz

def divergence_periodic(gx, gy, gz, voxel_size):
    """Negative divergence with periodic boundaries (adjoint of gradient)"""
    dx, dy, dz = voxel_size
    div_x = (gx - np.roll(gx, 1, axis=0)) / dx
    div_y = (gy - np.roll(gy, 1, axis=1)) / dy
    div_z = (gz - np.roll(gz, 1, axis=2)) / dz
    return -(div_x + div_y + div_z)

def _shrink_update(u, d, threshold):
    """Combined shrink and dual update for TV-ADMM"""
    u = u + d  # u := u + grad_x
    z = np.sign(u) * np.maximum(np.abs(u) - threshold, 0)  # shrink
    u = u - z  # dual update
    d = 2*z - u  # precompute d = z - u + z
    return u, d

# --- TKD (Truncated K-space Division) from QSM.jl direct.jl ---
def tkd_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1), thr=0.15):
    """TKD dipole inversion - instant (2 FFTs)

    From QSM.jl/src/inversion/direct.jl lines 37-48, 237-241
    If |D| <= thr: use sign(D)/thr (clamped inverse)
    If |D| > thr: use 1/D (normal inverse)
    """
    print(f"TKD: threshold={thr}")
    shape = local_field.shape
    D = create_dipole_kernel(shape, voxel_size, bdir)

    # Create truncated inverse kernel
    abs_D = np.abs(D)
    # Avoid division by zero in the else branch
    D_safe = np.where(abs_D > 0, D, 1.0)
    iD = np.where(abs_D <= thr,
                  np.sign(D) / thr,  # Clamped inverse (preserves sign)
                  1.0 / D_safe)  # Normal inverse
    iD[0, 0, 0] = 0  # DC term

    # Apply inversion: chi = IFFT(iD * FFT(f))
    f = local_field * mask
    F = np.fft.fftn(f)
    chi = np.real(np.fft.ifftn(iD * F)) * mask

    print("TKD completed")
    return chi

# --- Tikhonov Regularization from QSM.jl direct.jl ---
def tikh_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
             lambda_=0.01, reg='identity'):
    """Tikhonov dipole inversion - instant (2-3 FFTs)

    From QSM.jl/src/inversion/direct.jl lines 100-154, 248-270
    argmin_chi ||D*chi - f||^2 + (lambda/2)||Gamma*chi||^2

    Closed-form: chi_hat = D / (|D|^2 + lambda*Gamma) * f_hat
    """
    print(f"Tikhonov: lambda={lambda_}, reg={reg}")
    shape = local_field.shape
    D = create_dipole_kernel(shape, voxel_size, bdir)
    D_sq = D * D

    if reg == 'identity':
        # iD = D / (D^2 + lambda)
        iD = D / (D_sq + lambda_)
    elif reg == 'gradient':
        # iD = D / (D^2 + lambda*L) where L = negative Laplacian
        L = create_laplacian_kernel(shape, voxel_size)
        denom = D_sq + lambda_ * L
        iD = np.where(np.abs(denom) > 1e-12, D / denom, 0.0)
    elif reg == 'laplacian':
        # iD = D / (D^2 + lambda*L^2)
        L = create_laplacian_kernel(shape, voxel_size)
        L_sq = L * L
        denom = D_sq + lambda_ * L_sq
        iD = np.where(np.abs(denom) > 1e-12, D / denom, 0.0)
    else:
        raise ValueError(f"Unknown regularization: {reg}")

    iD[0, 0, 0] = 0  # DC term

    # Apply inversion
    f = local_field * mask
    F = np.fft.fftn(f)
    chi = np.real(np.fft.ifftn(iD * F)) * mask

    print("Tikhonov completed")
    return chi

# --- TV-ADMM from QSM.jl tv.jl ---
def tv_admm_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
                lambda_=0.001, rho=None, tol=0.001, maxit=250):
    """TV-ADMM dipole inversion

    From QSM.jl/src/inversion/tv.jl lines 56-382
    minimize: (1/2)||D*chi - f||^2 + lambda*||grad(chi)||_1

    Uses ADMM with precomputed frequency-domain operators.
    """
    if rho is None:
        rho = 100 * lambda_  # Default from QSM.jl

    print(f"TV-ADMM: lambda={lambda_}, rho={rho}, tol={tol}, maxit={maxit}")
    shape = local_field.shape
    D = create_dipole_kernel(shape, voxel_size, bdir)
    L = create_laplacian_kernel(shape, voxel_size)

    # Precompute frequency-domain operators (from QSM.jl tv.jl lines 220-235)
    D_conj = np.conj(D)
    D_sq = D_conj * D
    denom = D_sq + rho * L
    denom[denom == 0] = 1e-12
    iA = 1.0 / denom

    # Precompute RHS contribution: F_hat = iA * D' * FFT(f)
    f = local_field * mask
    F = np.fft.fftn(f)
    F_hat = iA * D_conj * F

    # Initialize ADMM variables
    x = np.zeros(shape, dtype=np.float64)
    ux, uy, uz = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    dx, dy, dz = gradient_periodic(x, voxel_size)

    lambda_rho = lambda_ / rho

    for iteration in range(maxit):
        x_prev = x.copy()

        # x-subproblem: frequency domain solve (QSM.jl tv.jl lines 260-273)
        div_d = divergence_periodic(dx, dy, dz, voxel_size)
        X = rho * iA * np.fft.fftn(div_d) + F_hat
        x = np.real(np.fft.ifftn(X))

        # Convergence check (QSM.jl tv.jl lines 276-293)
        diff = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-12)

        try:
            js_qsm_progress(iteration + 1, maxit)
        except:
            pass

        if diff < tol:
            print(f"  TV-ADMM converged at iteration {iteration + 1}")
            break

        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/{maxit}, diff = {diff:.2e}")

        # Compute gradients
        dx, dy, dz = gradient_periodic(x, voxel_size)

        # z-subproblem + dual update (QSM.jl tv.jl lines 295-349)
        ux, dx = _shrink_update(ux, dx, lambda_rho)
        uy, dy = _shrink_update(uy, dy, lambda_rho)
        uz, dz = _shrink_update(uz, dz, lambda_rho)

    print("TV-ADMM completed")
    return x * mask

def rts_qsm(local_field, mask, voxel_size, bdir=(0, 0, 1),
            delta=0.15, mu=1e5, rho=10.0, tol=1e-2, maxit=20):
    """RTS dipole inversion"""
    shape = local_field.shape
    D = create_dipole_kernel(shape, voxel_size, bdir)
    L = create_laplacian_kernel(shape, voxel_size)
    well_conditioned = np.abs(D) > delta
    M = np.where(well_conditioned, mu, 0.0)
    f = local_field * mask
    F = np.fft.fftn(f)

    print("Step 1: Direct inversion...")
    D_sq = D * D
    D_sq_safe = np.where(D_sq > delta**2, D_sq, delta**2)
    X = D * F / D_sq_safe
    x = np.real(np.fft.ifftn(X)) * mask

    print("Step 2: ADMM with TV...")
    denominator = M + rho * L
    denominator[denominator == 0] = 1e-12
    iA = rho / denominator
    X = np.fft.fftn(x)
    F_const = M * X / denominator
    px, py, pz = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    def gradient(x):
        return np.roll(x,-1,0)-x, np.roll(x,-1,1)-x, np.roll(x,-1,2)-x
    def divergence(px, py, pz):
        return (px-np.roll(px,1,0)) + (py-np.roll(py,1,1)) + (pz-np.roll(pz,1,2))
    def shrink(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)

    inv_rho = 1.0 / rho
    x_prev = x.copy()
    for iteration in range(maxit):
        try: js_qsm_progress(iteration + 1, maxit)
        except: pass
        gx, gy, gz = gradient(x)
        yx, yy, yz = shrink(gx+px, inv_rho), shrink(gy+py, inv_rho), shrink(gz+pz, inv_rho)
        px, py, pz = px+gx-yx, py+gy-yy, pz+gz-yz
        div_v = divergence(yx-px, yy-py, yz-pz)
        X = iA * np.fft.fftn(div_v) + F_const
        x = np.real(np.fft.ifftn(X))
        diff = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-12)
        if diff < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break
        x_prev = x.copy()
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}/{maxit}, diff = {diff:.2e}")
    return x * mask

# MEDI helper functions
def fgrad(chi, voxel_size):
    dx, dy, dz = voxel_size
    grad = np.zeros((*chi.shape, 3), dtype=chi.dtype)
    grad[:-1,:,:,0] = (chi[1:,:,:] - chi[:-1,:,:]) / dx
    grad[:,:-1,:,1] = (chi[:,1:,:] - chi[:,:-1,:]) / dy
    grad[:,:,:-1,2] = (chi[:,:,1:] - chi[:,:,:-1]) / dz
    return grad

def bdiv(grad_field, voxel_size):
    dx, dy, dz = voxel_size
    gx, gy, gz = grad_field[:,:,:,0], grad_field[:,:,:,1], grad_field[:,:,:,2]
    div = np.zeros_like(gx)
    div[0,:,:] = gx[0,:,:]/dx; div[1:-1,:,:] += (gx[1:-1,:,:]-gx[:-2,:,:])/dx; div[-1,:,:] += -gx[-2,:,:]/dx
    div[:,0,:] += gy[:,0,:]/dy; div[:,1:-1,:] += (gy[:,1:-1,:]-gy[:,:-2,:])/dy; div[:,-1,:] += -gy[:,-2,:]/dy
    div[:,:,0] += gz[:,:,0]/dz; div[:,:,1:-1] += (gz[:,:,1:-1]-gz[:,:,:-2])/dz; div[:,:,-1] += -gz[:,:,-2]/dz
    return div

def gradient_mask(magnitude, mask, percentage=0.9):
    gy, gx, gz = np.gradient(magnitude)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    masked_grad = grad_mag[mask > 0]
    if len(masked_grad) == 0: return np.ones_like(magnitude)
    threshold = np.percentile(masked_grad, percentage * 100)
    wG = np.ones_like(magnitude)
    wG[grad_mag > threshold] = 0
    return wG * mask

def cg_solve(A_op, b, tol=0.01, max_iter=100, precond=None):
    """Conjugate gradient solver with optional preconditioner"""
    x = np.zeros_like(b)
    r = b.copy()
    b_norm = np.sqrt(np.sum(b * b))
    if b_norm < 1e-12: return x

    if precond is not None:
        # Preconditioned CG
        z = precond(r)
        p = z.copy()
        rz_old = np.sum(r * z)
        for i in range(max_iter):
            Ap = A_op(p)
            pAp = np.sum(p * Ap)
            if np.abs(pAp) < 1e-12: break
            alpha = rz_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            if np.sqrt(np.sum(r * r)) / b_norm < tol: break
            z = precond(r)
            rz_new = np.sum(r * z)
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new
        return x
    else:
        # Standard CG (no preconditioner)
        p = r.copy()
        rsold = np.sum(r * r)
        for i in range(max_iter):
            Ap = A_op(p)
            pAp = np.sum(p * Ap)
            if np.abs(pAp) < 1e-12: break
            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.sum(r * r)
            if np.sqrt(rsnew) / b_norm < tol: break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

def medi_l1(local_field, mask, magnitude, voxel_size, lambda_=1000, max_iter=10,
            cg_max_iter=100, cg_tol=0.01, edge_percent=0.9, merit=False):
    """MEDI L1 dipole inversion - fully optimized (rfftn + preconditioner + float32)"""
    shape = local_field.shape
    eps = np.float32(1e-6)
    two_lambda = np.float32(2 * lambda_)

    # Use rfft dipole kernel in float32 (half-spectrum, ~2x faster FFTs)
    D_r = create_dipole_kernel_rfft(shape, voxel_size).astype(np.float32)

    # Precompute diagonal preconditioner: P = 1 / (eps + 2*lambda*|D|^2)
    # This approximates the inverse of the dominant fidelity term
    D_sq = D_r * D_r
    P_diag = (1.0 / (eps + two_lambda * D_sq)).astype(np.float32)
    print(f"MEDI (optimized): preconditioner range [{P_diag.min():.2e}, {P_diag.max():.2e}]")

    # Precompute mask and target in float32/complex64
    m = mask.astype(np.float32)
    b0 = m * np.exp(1j * local_field.astype(np.float32)).astype(np.complex64)

    # Precompute edge weighting (constant across all iterations)
    wG = gradient_mask(magnitude, mask, edge_percent).astype(np.float32)
    wG_sq = wG * wG

    chi = np.zeros(shape, dtype=np.float32)
    print(f"MEDI (rfftn+precond+f32): lambda={lambda_}, max_iter={max_iter}, cg_max_iter={cg_max_iter}")

    # Helper for rfft-based dipole convolution (float32)
    def Dconv_r(x):
        return np.fft.irfftn(D_r * np.fft.rfftn(x), s=shape).astype(np.float32)

    # Preconditioner: apply diagonal precond in Fourier space
    def precond(x):
        return np.fft.irfftn(P_diag * np.fft.rfftn(x), s=shape).astype(np.float32)

    for outer_iter in range(max_iter):
        try: js_qsm_progress(outer_iter + 1, max_iter)
        except: pass

        # Compute TV weights (changes each outer iteration)
        grad_chi = fgrad(chi, voxel_size)
        grad_mag_sq = wG_sq[:,:,:,np.newaxis] * grad_chi
        grad_mag_sq = np.sum(grad_mag_sq * grad_chi, axis=-1)
        Vr = (1.0 / np.sqrt(grad_mag_sq + eps)).astype(np.float32)
        Vr_wG_sq = Vr * wG_sq

        # Compute data fidelity weight (changes each outer iteration)
        phi_forward = Dconv_r(chi)
        w = (m * np.exp(1j * phi_forward)).astype(np.complex64)
        w_sq = np.real(np.conj(w) * w).astype(np.float32)

        # Define operators with precomputed weights
        def reg0(dx):
            g = fgrad(dx, voxel_size)
            g[:,:,:,0] *= Vr_wG_sq
            g[:,:,:,1] *= Vr_wG_sq
            g[:,:,:,2] *= Vr_wG_sq
            return bdiv(g, voxel_size).astype(np.float32)

        def fidelity(dx):
            Ddx = Dconv_r(dx)
            return Dconv_r(w_sq * Ddx)

        def A_op(dx): return reg0(dx) + two_lambda * fidelity(dx)

        # Compute RHS
        reg0_chi = reg0(chi)
        phase_term = np.imag(np.conj(w) * (w - b0)).astype(np.float32)
        rhs = reg0_chi + two_lambda * Dconv_r(phase_term)

        # Solve with preconditioned CG
        dx = cg_solve(A_op, -rhs, tol=cg_tol, max_iter=cg_max_iter, precond=precond)
        chi = chi + dx

        rel_change = np.linalg.norm(dx) / (np.linalg.norm(chi) + eps)
        print(f"  Iter {outer_iter + 1}/{max_iter}: rel_change={rel_change:.4f}")

        if rel_change < 0.1:
            print(f"  Converged at iteration {outer_iter + 1}")
            break

    # Return as float64 for compatibility with rest of pipeline
    return (chi * mask).astype(np.float64)

# Get voxel size
try:
    voxel_size = tuple(header_info.get_zooms()[:3])
except:
    voxel_size = (1.0, 1.0, 1.0)

print(f"Voxel size: {voxel_size} mm")
print(f"B0 field strength: ${magField} T")

# Run dipole inversion based on selected method
inv_method = dipole_method
if inv_method == 'tkd':
    print("Running TKD QSM dipole inversion (instant)...")
    tkd_thr = ${tkdSettings.threshold}
    print(f"TKD settings: threshold={tkd_thr}")
    qsm_result = tkd_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), thr=tkd_thr
    )
elif inv_method == 'tikhonov':
    print("Running Tikhonov QSM dipole inversion (instant)...")
    tikh_lambda = ${tikhonovSettings.lambda}
    tikh_reg = '${tikhonovSettings.reg}'
    print(f"Tikhonov settings: lambda={tikh_lambda}, reg={tikh_reg}")
    qsm_result = tikh_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), lambda_=tikh_lambda, reg=tikh_reg
    )
elif inv_method == 'tv':
    print("Running TV-ADMM QSM dipole inversion...")
    tv_lambda = ${tvSettings.lambda}
    tv_maxiter = ${tvSettings.maxIter}
    tv_tol = ${tvSettings.tol}
    print(f"TV-ADMM settings: lambda={tv_lambda}, maxiter={tv_maxiter}, tol={tv_tol}")
    qsm_result = tv_admm_qsm(
        local_fieldmap, processing_mask, voxel_size,
        bdir=(0, 0, 1), lambda_=tv_lambda, maxit=tv_maxiter, tol=tv_tol
    )
else:  # rts (default)
    print("Running RTS QSM dipole inversion...")
    rts_delta = ${rtsSettings.delta}
    rts_mu = ${rtsSettings.mu}
    rts_rho = ${rtsSettings.rho}
    rts_maxiter = ${rtsSettings.maxIter}
    print(f"RTS settings: delta={rts_delta}, mu={rts_mu}, rho={rts_rho}, maxiter={rts_maxiter}")
    qsm_result = rts_qsm(
        local_fieldmap, processing_mask, voxel_size,
        delta=rts_delta, mu=rts_mu, rho=rts_rho, maxit=rts_maxiter
    )

print(f"QSM result range: [{np.min(qsm_result[processing_mask]):.4f}, {np.max(qsm_result[processing_mask]):.4f}]")
print("Dipole inversion completed!")

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

// Post BET-specific progress
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
  const { magnitudeBuffer, voxelSize, betCode, fractionalIntensity, iterations, subdivisions } = data;
  const betIterations = iterations || 1000;
  const betSubdivisions = subdivisions || 4;

  try {
    // Initialize Pyodide if not already done
    if (!pyodide) {
      postBETLog("Initializing Pyodide for BET...");
      postBETProgress(0.05, 'Loading Pyodide...');

      importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.js');
      pyodide = await loadPyodide();

      postBETLog("Installing Python packages...");
      postBETProgress(0.1, 'Installing packages...');
      await pyodide.loadPackage(["numpy", "scipy", "micropip"]);

      postBETLog("Installing nibabel...");
      await pyodide.runPythonAsync(`
        import micropip
        await micropip.install("nibabel")
      `);
    }

    postBETProgress(0.15, 'Loading BET algorithm...');
    postBETLog("Loading BET algorithm...");

    // Load BET code
    await pyodide.runPython(betCode);

    postBETProgress(0.2, 'Loading magnitude data...');
    postBETLog("Loading magnitude data...");

    // Transfer data to Python
    pyodide.globals.set('mag_buffer', new Uint8Array(magnitudeBuffer));
    pyodide.globals.set('voxel_size', voxelSize);
    pyodide.globals.set('fractional_intensity', fractionalIntensity || 0.5);
    pyodide.globals.set('bet_iterations', betIterations);
    pyodide.globals.set('bet_subdivisions', betSubdivisions);

    // Set up progress callback for BET
    const betProgressCallback = (iteration, total) => {
      // BET runs from 20% to 90% of the progress bar
      const progress = 0.2 + (iteration / total) * 0.7;
      postBETProgress(progress, `BET: iteration ${iteration}/${total}`);
    };
    pyodide.globals.set('js_bet_progress', betProgressCallback);

    // Load and run BET
    await pyodide.runPython(`
import numpy as np
import nibabel as nib
from io import BytesIO

print("Loading magnitude image...")
mag_bytes = mag_buffer.to_py()
mag_fh = nib.FileHolder(BytesIO(mag_bytes))
mag_img = nib.Nifti1Image.from_file_map({'image': mag_fh, 'header': mag_fh})
mag_data = mag_img.get_fdata()

print(f"Image shape: {mag_data.shape}")
print(f"Voxel size from JS: {list(voxel_size.to_py())}")

# Get voxel size from header if available
try:
    header_voxel_size = tuple(mag_img.header.get_zooms()[:3])
    # Use header voxel size (it's in x, y, z order)
    vs = (header_voxel_size[2], header_voxel_size[1], header_voxel_size[0])  # Convert to z, y, x for Python
    print(f"Using header voxel size: {vs} mm (z, y, x)")
except:
    vs = tuple(voxel_size.to_py())
    print(f"Using provided voxel size: {vs} mm")

# Define progress callback wrapper
def progress_wrapper(iteration, total):
    try:
        js_bet_progress(iteration, total)
    except:
        pass

print(f"Running BET brain extraction (iterations={int(bet_iterations)}, subdivisions={int(bet_subdivisions)})...")
bet_mask = run_bet(
    mag_data,
    voxel_size=vs,
    fractional_intensity=float(fractional_intensity),
    iterations=int(bet_iterations),
    subdivisions=int(bet_subdivisions),
    progress_callback=progress_wrapper
)

print(f"BET mask shape: {bet_mask.shape}")
mask_count = np.sum(bet_mask > 0)
total_voxels = bet_mask.size
coverage_pct = (mask_count / total_voxels) * 100
print(f"Mask coverage: {mask_count}/{total_voxels} voxels ({coverage_pct:.1f}%)")

# Flatten mask for transfer - use Fortran order to match NIfTI convention
bet_mask_flat = bet_mask.flatten(order='F').astype(np.float32)
`);

    postBETProgress(0.95, 'Transferring mask...');
    postBETLog("Transferring mask data...");

    // Get the mask data
    const maskArray = pyodide.globals.get('bet_mask_flat').toJs();
    const coverage = pyodide.globals.get('coverage_pct');

    postBETProgress(1.0, 'Complete');
    postBETComplete(maskArray, `${coverage.toFixed(1)}%`);

  } catch (error) {
    postBETError(error.message);
    console.error('BET error:', error);
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

      case 'runBET':
        await runBET(data);
        break;

      default:
        postError(`Unknown message type: ${type}`);
    }
  } catch (error) {
    postError(error.message);
    console.error(error);
  }
};
