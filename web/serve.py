#!/usr/bin/env python3
"""Dev server for web/, stdlib only.

Sends COOP/COEP (cross-origin isolation) so the language model's threaded
CPU backend (pkg-lean-mt: SharedArrayBuffer, wasm threads) can run. Hugging
Face model fetches still work under COEP: they are CORS requests and HF
answers them with Access-Control-Allow-Origin. A host without these headers
(GitHub Pages) gets the single-thread CPU backend instead.

Usage: python3 web/serve.py [--port 8010] [--bind 127.0.0.1]
"""
import argparse
import http.server
import os
import socketserver
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm"}

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - %s\n" % (self.address_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    handler = lambda *a, **kw: Handler(*a, directory=ROOT, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {ROOT} on http://{args.bind}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
