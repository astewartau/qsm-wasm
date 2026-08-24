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

# Parse arguments. We build TWO wasm bundles:
#   - base (no onnx, ~2.3 MB): classical algorithms + chi-separation + relaxometry + the
#     deep-learning model registry. Loaded on page open.
#   - DL   (onnx, ~10.7 MB): adds tract-based deep-learning inference. Lazy-loaded in the
#     browser only when a deep-learning algorithm is selected (weights fetched in JS).
# `--simd` adds SIMD acceleration to both.
SIMD_FEAT=""
BUILD_TYPE="standard"
if [[ "$1" == "--simd" ]]; then
    SIMD_FEAT="simd"
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

# Threaded (multi-core) build via wasm-bindgen-rayon. Speeds up ALL rayon paths in qsm-core
# (classical algorithms) plus the tiled deep-learning loop. Requires nightly + build-std (to
# rebuild std with atomics) and a cross-origin-isolated page at runtime (COOP/COEP — see
# serve.py / the coi service worker). Disable with `--no-threads` for a single-threaded build
# that runs on any static host without special headers.
THREADS=1
for a in "$@"; do [[ "$a" == "--no-threads" ]] && THREADS=0; done

WP=(wasm-pack)
PAR_FEAT=""
if [[ "$THREADS" == "1" ]]; then
    if ! rustup toolchain list 2>/dev/null | grep -q nightly; then
        echo "Error: threaded build needs the nightly toolchain + rust-src:"
        echo "  rustup toolchain install nightly --component rust-src"
        echo "  (or build single-threaded with: ./build.sh --no-threads)"
        exit 1
    fi
    WP=(rustup run nightly wasm-pack)
    # Threaded wasm needs an *imported, shared* memory plus several linker symbols that recent
    # nightlies (>= 2026-05-06, rust #156174) stopped auto-exporting. Without these:
    #   • no --shared-memory/--import-memory  → memory stays non-shared (0x00) → initThreadPool
    #     fails at runtime with "WebAssembly.Memory could not be cloned";
    #   • missing --export=__heap_base/__wasm_init_tls/__tls_* → wasm-bindgen's thread transform
    #     panics ("failed to find __heap_base" / "__wasm_init_tls", "mem.import.is_some()").
    # 4 GiB is the wasm32 max-memory ceiling.
    THREAD_LINK="-C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296"
    THREAD_EXPORTS="-C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+atomics,+bulk-memory,+mutable-globals $THREAD_LINK $THREAD_EXPORTS"
    export CARGO_UNSTABLE_BUILD_STD="panic_abort,std"
    PAR_FEAT="parallel"
fi

# Comma-join non-empty feature names.
join_feats() { local IFS=,; echo "$*"; }
BASE_FEATS=$(join_feats ${SIMD_FEAT:+$SIMD_FEAT} ${PAR_FEAT:+$PAR_FEAT})
DL_FEATS=$(join_feats onnx ${SIMD_FEAT:+$SIMD_FEAT} ${PAR_FEAT:+$PAR_FEAT})

# Build WASM
echo "[1/4] Building WASM with wasm-pack..."
[[ -n "$SIMD_FEAT" ]] && echo "      SIMD acceleration enabled (Chrome 91+, Firefox 89+, Safari 16.4+)"
[[ "$THREADS" == "1" ]] && echo "      Threads ENABLED (wasm-bindgen-rayon; page must be cross-origin isolated)" \
                        || echo "      Threads disabled (single-threaded build)"
cd "$RUST_DIR"
echo "      Base bundle (classical + separation + relaxometry + model registry)..."
"${WP[@]}" build --target web --release --out-dir pkg ${BASE_FEATS:+--features "$BASE_FEATS"}
echo "      DL bundle (deep-learning inference via tract; lazy-loaded)..."
"${WP[@]}" build --target web --release --out-dir pkg-dl --out-name qsm_wasm_dl --features "$DL_FEATS"

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
# DL bundle (lazy-loaded)
cp "$RUST_DIR/pkg-dl/qsm_wasm_dl.js" "$WASM_DIR/"
cp "$RUST_DIR/pkg-dl/qsm_wasm_dl_bg.wasm" "$WASM_DIR/"
cp "$RUST_DIR/pkg-dl/qsm_wasm_dl.d.ts" "$WASM_DIR/" 2>/dev/null || true
cp "$RUST_DIR/pkg-dl/qsm_wasm_dl_bg.wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true

# Threaded builds emit a wasm-bindgen-rayon `snippets/` dir with the worker bootstrap JS, which
# qsm_wasm.js / qsm_wasm_dl.js import by relative path — copy it alongside (harmless if absent).
rm -rf "$WASM_DIR/snippets"
if [ -d "$RUST_DIR/pkg/snippets" ]; then cp -r "$RUST_DIR/pkg/snippets" "$WASM_DIR/"; fi
if [ -d "$RUST_DIR/pkg-dl/snippets" ]; then cp -r "$RUST_DIR/pkg-dl/snippets/." "$WASM_DIR/snippets/"; fi

# wasm-bindgen-rayon's worker helper loads the main module via `import('../../..')`, which only
# resolves under a bundler; over raw HTTP (--target web) it hits the `/wasm/` *directory* and
# fails. Give each bundle its own patched helper that imports its explicit main JS. No-op for
# single-threaded builds (no snippets dir).
if [ -d "$WASM_DIR/snippets" ]; then
python3 - "$WASM_DIR" <<'PYEOF'
import os, sys
wasm = sys.argv[1]
snip = os.path.join(wasm, 'snippets')
hashdir = next((d for d in os.listdir(snip) if d.startswith('wasm-bindgen-rayon')), None)
if hashdir:
    src = open(os.path.join(snip, hashdir, 'src', 'workerHelpers.js')).read()
    for tag, mainjs in [('rayon-base', 'qsm_wasm.js'), ('rayon-dl', 'qsm_wasm_dl.js')]:
        if not os.path.exists(os.path.join(wasm, mainjs)):
            continue
        outdir = os.path.join(snip, tag); os.makedirs(outdir, exist_ok=True)
        open(os.path.join(outdir, 'workerHelpers.js'), 'w').write(
            src.replace("import('../../..')", f"import('../../{mainjs}')"))
        p = os.path.join(wasm, mainjs); js = open(p).read()
        open(p, 'w').write(js.replace(
            f'./snippets/{hashdir}/src/workerHelpers.js', f'./snippets/{tag}/workerHelpers.js'))
        print(f"  rayon worker helper patched: {mainjs} -> snippets/{tag}")
PYEOF
fi

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
