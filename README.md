# QSMbly: Browser-Based Quantitative Susceptibility Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Ready-green.svg)](https://pages.github.com/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Powered-blue.svg)](https://webassembly.org/)

A complete **Quantitative Susceptibility Mapping (QSM)** pipeline that runs entirely in your web browser using WebAssembly and Pyodide. No installation, no backend servers, no data uploads - just pure client-side MRI processing.

## 🌟 Features

- **🔒 Completely Private**: All processing happens locally in your browser - your data never leaves your computer
- **🚀 Zero Installation**: No Python, MATLAB, or specialized software required
- **📱 Cross-Platform**: Works on Windows, macOS, Linux, and even mobile devices
- **⚡ Interactive**: Real-time visualization with NiiVue, adjustable contrast, and masking thresholds
- **💾 Portable**: Static files can be hosted anywhere (GitHub Pages, local server, etc.)
- **🔬 Complete Pipeline**: Phase unwrapping → Background removal → Dipole inversion

## 🏗️ QSM Pipeline Steps

### 1. **Phase Scaling** 
- **Purpose**: Normalizes phase values to standard [-π, +π] range
- **Input**: Raw phase data (any range, e.g., -4096 to +4095)
- **Output**: Phase data scaled to [-π, +π] in float32 format
- **Robustness**: Handles integer or float input data with any min/max range

### 2. **Phase Unwrapping**
- **Algorithm**: Laplacian-based unwrapping
- **Input**: Scaled phase image in [-π, +π] range
- **Output**: Unwrapped fieldmap in Hz

### 3. **Background Field Removal**
- **Algorithm**: SHARP (Sophisticated Harmonic Artifact Reduction for Phase data)
- **Purpose**: Removes background field contributions from sources outside the brain
- **Output**: Local tissue fieldmap

### 4. **Dipole Inversion**
- **Algorithm**: RTS (Rapid Two-Step) method
- **Purpose**: Converts local fieldmap to magnetic susceptibility values
- **Output**: Quantitative susceptibility map (χ-map)

### 5. **Brain Masking**
- **Method**: Intensity-based thresholding with morphological operations
- **Interactive**: Adjustable threshold with real-time preview
- **Purpose**: Define brain tissue regions for processing

## 🚀 Quick Start

### Option 1: GitHub Pages (Recommended)
1. Visit the live demo: `https://yourusername.github.io/qsmbly/`
2. Upload your magnitude and phase NIfTI files
3. Set acquisition parameters (Echo Time, Field Strength)
4. Run the pipeline!

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/qsmbly.git
cd qsmbly

# Serve locally (Python)
python -m http.server 8080

# Or use Node.js
npx serve .

# Open http://localhost:8080
```

## 📁 Repository Structure

```
qsmbly/
├── index.html              # Main application interface
├── js/
│   └── qsm-app.js          # Main application logic
├── css/
│   └── modern-styles.css   # Modern UI styling
├── python/                 # QSM processing algorithms
│   ├── masking3.py         # Brain masking
│   ├── unwrap.py           # Phase unwrapping
│   ├── bg_removal_sharp.py # Background field removal
│   └── rts_wasm_standard.py# Dipole inversion
├── assets/                 # Sample data and masks
│   ├── magnitude.nii       # Example magnitude image
│   ├── phase.nii          # Example phase image
│   └── MNI_mask.nii       # Standard brain mask
├── demo/                   # Demo datasets
├── Benchmark/              # Validation results
└── settings.json          # Default acquisition parameters
```

## 📊 Input Requirements

### Required Files
- **Magnitude Image**: `.nii` or `.nii.gz` format
- **Phase Image**: `.nii` or `.nii.gz` format (same dimensions as magnitude)

### Acquisition Parameters
- **Echo Time (TE)**: In seconds (e.g., 0.004 for 4ms)
- **Magnetic Field Strength**: In Tesla (e.g., 3.0, 7.0)

### Recommended Acquisition
- **Sequence**: 3D Gradient Echo (GRE) or FLASH
- **Resolution**: ≤ 1mm³ isotropic
- **Field Strength**: 3T or 7T
- **Echo Time**: First echo of multi-echo sequence

## 🎮 Usage Guide

### Step 1: Upload Data
1. Click "Choose magnitude file" and select your magnitude NIfTI
2. Click "Choose phase file" and select your phase NIfTI
3. (Optional) Load settings from JSON file

### Step 2: Set Parameters
1. Enter Echo Time in seconds
2. Enter Magnetic Field Strength in Tesla
3. Click "Visualize" buttons to preview your data

### Step 3: Create Brain Mask
1. Click "Run QSM Pipeline"
2. Adjust masking threshold using the slider
3. Preview the mask in real-time
4. Click "Use This Mask" when satisfied

### Step 4: Process QSM
The pipeline will automatically run:
- **Phase Scaling** (~5 seconds) - Normalizes phase to [-π, +π] range
- **Phase Unwrapping** (~10 seconds)
- **Background Removal** (~30 seconds)  
- **Dipole Inversion** (~45 seconds)

### Step 5: Explore Results
- Navigate between processing stages using buttons
- Adjust contrast for optimal visualization
- Download results or save screenshots

## 🔧 Technical Details

### WebAssembly Stack
- **Pyodide**: Python scientific computing in the browser
- **NumPy/SciPy**: Numerical processing
- **NiBabel**: NIfTI file handling

### Visualization
- **NiiVue**: High-performance WebGL neuroimaging viewer
- **Multi-planar views**: Axial, coronal, sagittal
- **Interactive contrast**: Real-time adjustment

### Browser Requirements
- **Modern Browser**: Chrome 88+, Firefox 79+, Safari 14+, Edge 88+
- **Memory**: 4GB RAM recommended for typical datasets
- **Storage**: ~100MB for Pyodide + dependencies

## 📈 Performance & Limitations

### Typical Processing Times
| Dataset Size | Phase Unwrapping | Background Removal | Dipole Inversion | Total |
|-------------|------------------|-------------------|------------------|-------|
| 256³        | ~10s            | ~30s             | ~45s            | ~1.5min |
| 512³        | ~30s            | ~2min            | ~3min           | ~5.5min |

### Current Limitations
- **Memory**: Limited by browser (typically 2-4GB)
- **Dataset Size**: Optimal for ≤ 512³ voxels
- **Processing Speed**: ~5-10x slower than native code
- **Mobile**: Limited by device memory and processing power

## 🔬 Validation & Benchmarks

The algorithms have been validated against reference implementations:

- **Phase Unwrapping**: Compared with MRITOOLS
- **Background Removal**: Validated against SHARP reference
- **Dipole Inversion**: Benchmarked against RTS Julia implementation

See `Benchmark/` directory for detailed validation results and error analysis.

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Performance**: Optimize algorithms for WebAssembly
- **Features**: Additional QSM methods (MEDI, iLSQR, etc.)
- **UI/UX**: Enhanced visualization and user experience
- **Validation**: More extensive benchmarking

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/qsmbly.git
cd qsmbly

# For Python algorithm development
pip install nibabel numpy scipy

# For frontend development
npm install -g live-server
live-server --port=8080
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NiiVue Team**: For the excellent neuroimaging viewer
- **Pyodide Project**: For bringing Python to the browser
- **QSM Community**: For developing and sharing algorithms
- **Original Developer**: For the foundational undergraduate work

## 📚 References

1. **Phase Unwrapping**: Schofield & Zhu. "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters* (2003)
2. **SHARP**: Schweser et al. "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase." *Magnetic Resonance in Medicine* (2011)
3. **RTS**: Kames et al. "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors." *NeuroImage* (2018)

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/qsmbly/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/qsmbly/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/qsmbly/wiki)

---

**Made with ❤️ for the neuroimaging community**