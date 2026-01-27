# QSM-WASM Functional Specification

This document describes the functionality of the QSM-WASM application for a developer implementing a new UI.

## Overview

QSM-WASM is a browser-based Quantitative Susceptibility Mapping (QSM) pipeline. It processes multi-echo MRI data entirely client-side using WebAssembly (Pyodide) and displays results using NiiVue.

## Core Dependencies

```html
<!-- Pyodide - Python in the browser -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.js"></script>

<!-- NiiVue - 3D medical image viewer -->
<script type="module">
  import { Niivue, NVImage } from "https://unpkg.com/@niivue/niivue@0.57.0/dist/index.js"
  window.Niivue = Niivue;
  window.NVImage = NVImage;
</script>
```

---

## User Inputs

### File Inputs

Users must provide three types of files:

| Input | Format | Quantity | Purpose |
|-------|--------|----------|---------|
| Magnitude images | `.nii` or `.nii.gz` | 1 per echo | MRI signal intensity |
| Phase images | `.nii` or `.nii.gz` | 1 per echo | MRI phase data |
| JSON metadata | `.json` | 1 per echo | Contains echo times (BIDS format) |

**Validation rules:**
- Number of magnitude files must equal number of phase files
- At least one echo time must be extracted from JSON files
- All files for a given echo should have matching dimensions (validated during processing)

**JSON echo time extraction:**
The app looks for echo time in these fields (in order):
1. `EchoTime` (seconds, converted to ms)
2. `echo_time` (seconds, converted to ms)
3. `TE` (already in ms)

### Parameters

| Parameter | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| Field Strength | number | 7.0 | Tesla | Main magnetic field strength of the MRI scanner |
| Unwrap Mode | select | "individual" | - | Phase unwrapping strategy |

**Unwrap Mode options:**
- `individual` - Unwrap each echo independently (recommended)
- `temporal` - Use temporal information across echoes

---

## Processing Pipeline

The pipeline runs these steps sequentially:

### 1. Pyodide Initialization (one-time)
- Load Pyodide runtime
- Install packages: `numpy`, `scipy`, `micropip`
- Install `nibabel` via micropip
- Load ROMEO algorithm from `./romeo_python.py`

### 2. Load Multi-Echo Data
- Read all magnitude and phase NIfTI files
- Stack into 4D arrays (x, y, z, echo)
- Scale phase to [-π, π] if values exceed expected range
- Store affine matrix and header from first echo

### 3. ROMEO Phase Unwrapping
- Unwrap phase using ROMEO algorithm
- Calculate B0 fieldmap from multi-echo data
- Generate processing mask

### 4. Background Field Removal
- V-SHARP algorithm
- Removes field contributions from outside the brain
- Outputs local fieldmap

### 5. Dipole Inversion
- RTS (Rapid Two-Step) algorithm
- Converts local fieldmap to susceptibility values
- Outputs QSM result in ppm

---

## Pipeline Outputs (Stages)

Each stage produces a 3D volume that can be viewed and downloaded:

| Stage ID | Name | Description |
|----------|------|-------------|
| `magnitude` | Magnitude | First echo magnitude image |
| `phase` | Phase | First echo phase image |
| `mask` | Mask | Binary processing mask |
| `B0` | B0 Field | B0 fieldmap in Hz |
| `bgRemoved` | Local Field | Local fieldmap after background removal |
| `final` | QSM | Final susceptibility map in ppm |

**Download format:** NIfTI (`.nii`)
**Download filename pattern:** `{stage}_{YYYY-MM-DD}.nii`

---

## Progress Reporting

The app reports progress at these points:

| Progress | Stage |
|----------|-------|
| 5-7% | Pyodide initialization |
| 10% | Starting pipeline |
| 30% | Data loaded, starting unwrapping |
| 60% | Unwrapping complete, starting background removal |
| 80% | Background removal complete, starting inversion |
| 100% | Complete |

Progress should also display a text status message.

---

## Console/Log Output

The app outputs timestamped messages for:
- Initialization status
- File loading confirmation
- Each pipeline step start/completion
- Errors with descriptive messages

Format: `[HH:MM:SS] Message text`

---

## NiiVue Viewer Component

NiiVue displays 3D medical images. Here's how to use it:

### Setup

```javascript
// Create instance with location callback
const nv = new Niivue({
  onLocationChange: (data) => {
    // data.string contains formatted coordinate/intensity info
    // Example: "x=128 y=128 z=64 value=0.42"
  }
});

// Attach to a canvas element
await nv.attachTo("canvasElementId");

// Configure display
nv.setMultiplanarPadPixels(5);  // Padding between slice views
nv.setSliceType(nv.sliceTypeMultiplanar);  // Show axial, coronal, sagittal
```

### Required HTML

```html
<canvas id="gl1"></canvas>
```

The canvas should be sized via CSS. NiiVue will fill its container.

### Loading Volumes

```javascript
// From a File object
const arrayBuffer = await file.arrayBuffer();
const volume = nv.loadFromFile({
  file: file,
  buffer: arrayBuffer
});

// Replace current volume
nv.removeVolumeByIndex(0);
await nv.addVolume(volume);
```

### Slice Types

```javascript
nv.setSliceType(nv.sliceTypeMultiplanar);  // 3-plane view (default)
nv.setSliceType(nv.sliceTypeAxial);        // Single axial slice
nv.setSliceType(nv.sliceTypeCoronal);      // Single coronal slice
nv.setSliceType(nv.sliceTypeSagittal);     // Single sagittal slice
nv.setSliceType(nv.sliceTypeRender);       // 3D rendering
```

### Contrast/Window Adjustment

```javascript
// Get current volume
const volume = nv.volumes[0];

// Set intensity range (cal_min, cal_max)
volume.cal_min = minValue;
volume.cal_max = maxValue;
nv.updateGLVolume();

// Or use normalized values (0-1)
nv.setCalMinMax(0, volumeIndex, minValue, maxValue);
```

### Colormap

```javascript
// Set colormap for a volume
nv.setColormap(volumeIndex, colormapName);

// Common colormaps: "gray", "hot", "cool", "viridis", "plasma"
```

### Mouse Interaction

NiiVue handles mouse interaction automatically:
- **Click/drag**: Navigate through slices
- **Scroll**: Zoom or scroll through slices
- **Right-drag**: Adjust window/level (contrast)

The `onLocationChange` callback fires when the cursor position changes, providing coordinate and intensity data.

### Saving/Export

```javascript
// Save screenshot as PNG
nv.saveScene("screenshot.png");

// Get the current volume as NIfTI bytes (if needed)
// Typically handled via the results object in this app
```

---

## Application State

### File Storage Structure

```javascript
multiEchoFiles = {
  magnitude: [{ file: File, name: string }, ...],
  phase: [{ file: File, name: string }, ...],
  json: [{ file: File, name: string }, ...],
  echoTimes: [{ file: string, echoTime: number, json: object }, ...]
}
```

### Results Storage

```javascript
results = {
  magnitude: { path: string, file: File },
  phase: { path: string, file: File },
  mask: { path: string, file: File },
  B0: { path: string, file: File },
  bgRemoved: { path: string, file: File },
  final: { path: string, file: File }
}
```

---

## User Actions

| Action | Trigger | Preconditions | Result |
|--------|---------|---------------|--------|
| Upload magnitude files | File input change | None | Files stored, list updated |
| Upload phase files | File input change | None | Files stored, list updated |
| Upload JSON files | File input change | None | Files parsed, echo times extracted |
| Remove file | Click remove button | Files loaded | File removed from list |
| Preview magnitude | Button click | ≥1 magnitude file | First echo shown in viewer |
| Preview phase | Button click | ≥1 phase file | First echo shown in viewer |
| Run pipeline | Button click | Valid files + parameters | Pipeline executes |
| View stage | Button/tab click | Pipeline complete | Stage loaded in viewer |
| Download stage | Button click | Pipeline complete | NIfTI file downloaded |

---

## Validation States

The "Run Pipeline" action should be disabled until all conditions are met:

```javascript
const canRun = (
  magnitudeFiles.length > 0 &&
  phaseFiles.length > 0 &&
  magnitudeFiles.length === phaseFiles.length &&
  echoTimes.length > 0 &&
  fieldStrength > 0
);
```

Display appropriate feedback when:
- Files are mismatched (different counts)
- Echo times are missing
- Field strength is invalid

---

## Error Handling

Errors can occur at these points:
- File parsing (invalid NIfTI or JSON)
- Pyodide initialization failure
- Processing errors (dimension mismatch, numerical issues)
- NiiVue loading errors

All errors should be:
1. Logged to console
2. Displayed to user with descriptive message
3. Progress reset to indicate failure

---

## File Structure

```
qsm-wasm/
├── index.html           # Main HTML
├── css/
│   └── modern-styles.css
├── js/
│   └── qsm-app-romeo.js # Main application logic
├── romeo_python.py      # ROMEO algorithm (loaded at runtime)
└── python/              # Additional Python modules (if any)
```

The application is a static site - no build step required. Serve with any HTTP server.
