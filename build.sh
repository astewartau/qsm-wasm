#!/bin/bash
# Build script for QSMbly WebAssembly components
# This compiles the Rust code to WASM and copies it to the serve directory
#
# Usage:
#   ./build.sh           # Standard build (maximum browser compatibility)
#   ./build.sh --simd    # SIMD-accelerated build (faster, requires modern browsers)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust-wasm"
WASM_DIR="$SCRIPT_DIR/wasm"

# Parse arguments. `onnx` is always on — it provides the deep-learning models (their
# weights are fetched in JS, not by the wasm). `--simd` adds SIMD acceleration.
FEATURES="onnx"
BUILD_TYPE="standard"
if [[ "$1" == "--simd" ]]; then
    FEATURES="onnx,simd"
    BUILD_TYPE="SIMD-accelerated"
fi

echo "=== QSMbly WASM Build ($BUILD_TYPE) ==="
echo ""

# Check for required tools
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Install it with: cargo install wasm-pack"
    echo "Or visit: https://rustwasm.github.io/wasm-pack/installer/"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "Error: cargo (Rust) is not installed."
    echo "Install from: https://rustup.rs/"
    exit 1
fi

# Build WASM
echo "[1/4] Building WASM with wasm-pack..."
if [[ -n "$SIMD_FLAG" ]]; then
    echo "      SIMD acceleration enabled (requires Chrome 91+, Firefox 89+, Safari 16.4+)"
fi
cd "$RUST_DIR"
wasm-pack build --target web --release --features "$FEATURES"

echo ""
echo "[2/4] Generating algorithm defaults from QSM.rs..."
cd "$SCRIPT_DIR"
node scripts/generate-defaults.mjs

echo ""
echo "[3/4] Copying WASM files to serve directory..."
cp "$RUST_DIR/pkg/qsm_wasm.js" "$WASM_DIR/"
cp "$RUST_DIR/pkg/qsm_wasm_bg.wasm" "$WASM_DIR/"
cp "$RUST_DIR/pkg/qsm_wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
cp "$RUST_DIR/pkg/qsm_wasm_bg.wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true

# Copy romeo files if they exist
if [ -f "$RUST_DIR/pkg/romeo_wasm.js" ]; then
    cp "$RUST_DIR/pkg/romeo_wasm.js" "$WASM_DIR/"
    cp "$RUST_DIR/pkg/romeo_wasm_bg.wasm" "$WASM_DIR/"
    cp "$RUST_DIR/pkg/romeo_wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
    cp "$RUST_DIR/pkg/romeo_wasm_bg.wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
fi

echo ""
echo "[4/5] Downloading example data from OSF..."
DATA_DIR="$SCRIPT_DIR/data/example"
if [ -d "$DATA_DIR" ] && [ "$(ls -1 "$DATA_DIR"/*.nii.gz 2>/dev/null | wc -l)" -eq 8 ]; then
    echo "      Example data already exists, skipping download."
else
    mkdir -p "$DATA_DIR"
    OSF_BASE="https://files.au-1.osf.io/v1/resources/z79k5/providers/osfstorage"
    curl -sL -o "$DATA_DIR/sub-1_echo-1_part-mag_MEGRE.json"      "$OSF_BASE/6a031592f1ede34d5380d7bf"
    curl -sL -o "$DATA_DIR/sub-1_echo-1_part-mag_MEGRE.nii.gz"    "$OSF_BASE/6a0315d064a982ca5fec727c"
    curl -sL -o "$DATA_DIR/sub-1_echo-1_part-phase_MEGRE.json"    "$OSF_BASE/6a031594d9869f43beec7017"
    curl -sL -o "$DATA_DIR/sub-1_echo-1_part-phase_MEGRE.nii.gz"  "$OSF_BASE/6a0315d369675bd488fdf827"
    curl -sL -o "$DATA_DIR/sub-1_echo-2_part-mag_MEGRE.json"      "$OSF_BASE/6a031594f1ede34d5380d7c2"
    curl -sL -o "$DATA_DIR/sub-1_echo-2_part-mag_MEGRE.nii.gz"    "$OSF_BASE/6a0315d469675bd488fdf828"
    curl -sL -o "$DATA_DIR/sub-1_echo-2_part-phase_MEGRE.json"    "$OSF_BASE/6a031591ca0aa1330880d636"
    curl -sL -o "$DATA_DIR/sub-1_echo-2_part-phase_MEGRE.nii.gz"  "$OSF_BASE/6a0315d369675bd488fdf825"
    curl -sL -o "$DATA_DIR/sub-1_echo-3_part-mag_MEGRE.json"      "$OSF_BASE/6a03159071aec37958ec71a4"
    curl -sL -o "$DATA_DIR/sub-1_echo-3_part-mag_MEGRE.nii.gz"    "$OSF_BASE/6a0315d37bd2b1503380d819"
    curl -sL -o "$DATA_DIR/sub-1_echo-3_part-phase_MEGRE.json"    "$OSF_BASE/6a0315917bd2b1503380d7d1"
    curl -sL -o "$DATA_DIR/sub-1_echo-3_part-phase_MEGRE.nii.gz"  "$OSF_BASE/6a0315d542632a6310ec7522"
    curl -sL -o "$DATA_DIR/sub-1_echo-4_part-mag_MEGRE.json"      "$OSF_BASE/6a03159442632a6310ec74a2"
    curl -sL -o "$DATA_DIR/sub-1_echo-4_part-mag_MEGRE.nii.gz"    "$OSF_BASE/6a0315d5a6ee1f1cc6fdf838"
    curl -sL -o "$DATA_DIR/sub-1_echo-4_part-phase_MEGRE.json"    "$OSF_BASE/6a03159442632a6310ec74a1"
    curl -sL -o "$DATA_DIR/sub-1_echo-4_part-phase_MEGRE.nii.gz"  "$OSF_BASE/6a0315d571aec37958ec71c0"
    echo "      Downloaded $(ls -1 "$DATA_DIR" | wc -l) files."
fi

echo ""
echo "[5/5] Build complete!"
echo ""
echo "WASM files in $WASM_DIR:"
ls -lh "$WASM_DIR"/*.wasm "$WASM_DIR"/*.js 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "To start the development server:"
echo "  python -m http.server 8080"
echo "  # Then open http://localhost:8080"
