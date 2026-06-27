"""NeuroZip command line — ffmpeg-like front end.

This skeleton parses commands and wires them to the (currently stubbed) library so that
`neurozip --help` works from day one. Subcommands raise NotImplementedError until their
backing stages are built (see instructions.md, gitignored).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _add_compress(sub):
    p = sub.add_parser("compress", help="Compress a file or directory of EEG epochs.")
    p.add_argument("input", help="Path to an EEG file or directory.")
    p.add_argument("-o", "--output", required=True, help="Output .nz path.")
    p.add_argument("--ratio", type=float, default=50.0, help="Target compression ratio.")
    return p


def _add_decompress(sub):
    p = sub.add_parser("decompress", help="Decompress a .nz archive back to EEG epochs.")
    p.add_argument("input", help="Path to a .nz archive.")
    p.add_argument("-o", "--output", required=True, help="Output directory.")
    return p


def _add_search(sub):
    p = sub.add_parser("search", help="Text/image query -> retrieve stored epochs.")
    p.add_argument("query", help="A text query, e.g. 'accordion'.")
    p.add_argument("--in", dest="archive", required=True, help="Path to a .nz archive.")
    p.add_argument("--topk", type=int, default=5, help="Number of results.")
    return p


def _add_ui(sub):
    sub.add_parser("ui", help="Launch the drag-and-drop interface.")


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
    _add_ui(sub)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    # Subcommands are intentionally not wired to real implementations yet.
    raise NotImplementedError(
        f"`neurozip {args.command}` is not implemented yet — see instructions.md."
    )


if __name__ == "__main__":
    sys.exit(main())
