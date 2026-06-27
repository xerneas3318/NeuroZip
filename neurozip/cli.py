"""NeuroZip command line — ffmpeg-like front end.

Beta status: `compress` / `decompress` emit placeholder (random) results so the whole
pipeline and packaging can be exercised end to end before the real codec lands. The
local UI (`serve` / `ui`) is fully functional for browsing + drag-and-drop folder
conversion against the placeholder backend.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from . import __version__
from . import core


def _add_compress(sub):
    p = sub.add_parser("compress", help="Compress a file or directory of epochs.")
    p.add_argument("input", help="Path to a file or directory.")
    p.add_argument("-o", "--output", help="Output .nz path (default: <input>.nz).")
    p.add_argument("--ratio", type=float, default=50.0, help="Target compression ratio.")
    return p


def _add_decompress(sub):
    p = sub.add_parser("decompress", help="Decompress a .nz archive.")
    p.add_argument("input", help="Path to a .nz archive.")
    p.add_argument("-o", "--output", help="Output directory.")
    return p


def _add_search(sub):
    p = sub.add_parser("search", help="Text/image query -> retrieve stored epochs.")
    p.add_argument("query", help="A text query, e.g. 'accordion'.")
    p.add_argument("--in", dest="archive", required=True, help="Path to a .nz archive.")
    p.add_argument("--topk", type=int, default=5, help="Number of results.")
    return p


def _add_serve(sub):
    for name, helptext in (
        ("serve", "Start the local NeuroZip UI (drag-and-drop folder convert)."),
        ("ui", "Alias for `serve`."),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--port", type=int, default=7878, help="Port (default 7878).")
        p.add_argument("--host", default="127.0.0.1", help="Bind host (default localhost).")
        p.add_argument("--no-browser", action="store_true", help="Do not open a browser.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurozip",
        description="Task-aware neural compression for brain/biomedical signals.",
    )
    parser.add_argument("--version", action="version", version=f"neurozip {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_compress(sub)
    _add_decompress(sub)
    _add_search(sub)
    _add_serve(sub)
    return parser


def _cmd_compress(args) -> int:
    out = args.output or (args.input.rstrip(os.sep) + ".nz")
    result = core.compress(args.input, out, ratio=args.ratio)
    print(
        f"[neurozip:beta] compressed {result['n_items']} item(s) "
        f"{result['original_bytes']:,} B -> {result['compressed_bytes']:,} B "
        f"({result['ratio']:.1f}x)  ->  {out}"
    )
    print("  note: beta placeholder — figures are randomized until the real codec ships.")
    return 0


def _cmd_decompress(args) -> int:
    out = args.output or os.path.splitext(args.input)[0] + "_restored"
    result = core.decompress(args.input, out)
    print(
        f"[neurozip:beta] decompressed {args.input} -> {out} "
        f"({result['n_items']} epoch(s), {result['restored_bytes']:,} B)"
    )
    print("  note: beta placeholder — output is synthetic until the real codec ships.")
    return 0


def _cmd_search(args) -> int:
    hits = core.search(args.query, args.archive, topk=args.topk)
    print(f"[neurozip:beta] top-{args.topk} for {args.query!r} in {args.archive}:")
    for rank, (concept, score) in enumerate(hits, 1):
        print(f"  {rank}. {concept:<20} score={score:.3f}")
    print("  note: beta placeholder — retrieval is randomized until the judge model loads.")
    return 0


def _cmd_serve(args) -> int:
    from . import server

    return server.run(host=args.host, port=args.port, open_browser=not args.no_browser)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    dispatch = {
        "compress": _cmd_compress,
        "decompress": _cmd_decompress,
        "search": _cmd_search,
        "serve": _cmd_serve,
        "ui": _cmd_serve,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
