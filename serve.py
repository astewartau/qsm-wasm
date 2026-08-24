#!/usr/bin/env python3
"""Static dev server that sets the cross-origin-isolation headers required for wasm threads.

`SharedArrayBuffer` (and therefore wasm-bindgen-rayon multithreading) only works on a
cross-origin-isolated page, which needs these two response headers on every request:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

Python's plain `http.server` doesn't send them, so use this instead:

    python3 serve.py [port]        # default 8080

For static production hosts that can't set headers (e.g. GitHub Pages), use the
`coi-serviceworker` shim in index.html instead — this server is for local dev.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class COIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        # `credentialless` (not `require-corp`) still enables SharedArrayBuffer but lets
        # cross-origin subresources (tagify/unpkg, Google fonts) load without CORP headers —
        # matching the coi-serviceworker config used on the deployed (GitHub Pages) site.
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # Dev convenience: never cache, so rebuilt wasm/JS is always picked up.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Serving cross-origin-isolated on http://localhost:{port}  (COOP/COEP + no-cache)")
    ThreadingHTTPServer(("", port), COIHandler).serve_forever()
