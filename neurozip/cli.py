"""NeuroZip command line - ffmpeg-like front end (real model).

`compress` / `decompress` / `embed` run the trained EEG codec + frozen CLIP-space
projector. `serve` launches the upload UI. `--help` and `--version` work without
torch installed; real commands require the `ml` extra and the model checkpoints
(`neurozip download`, or a local ./checkpoints).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _add(sub, name, help):
    return sub.add_parser(name, help=help)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurozip",
        description="Task-aware neural compression for brain/biomedical signals.",
    )
    p.add_argument("--version", action="version", version=f"neurozip {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    c = _add(sub, "compress", "Compress an EEG epoch file (.npy) with the trained codec.")
    c.add_argument("input", nargs="?", help="Path to a .npy of shape (63,250) or (N,63,250).")
    c.add_argument("--sample", type=int, metavar="IDX", help="Use a bundled demo epoch instead of a file.")
    c.add_argument("-o", "--output", help="Output .nz path (default: <input>.nz).")
    c.add_argument("--tier", default="high", choices=["low", "med", "high", "xhigh"])
    c.add_argument("--variant", default="neurozip", choices=["neurozip", "fidelity"])

    d = _add(sub, "decompress", "Reconstruct EEG from a .nz archive.")
    d.add_argument("input", help="Path to a .nz archive.")
    d.add_argument("-o", "--output", help="Output .npy (default: <input>.recon.npy).")

    e = _add(sub, "embed", "EEG epoch (.npy) -> 512-d CLIP-space embedding from the frozen judge.")
    e.add_argument("input", nargs="?", help="Path to a .npy epoch.")
    e.add_argument("--sample", type=int, metavar="IDX", help="Use a bundled demo epoch.")
    e.add_argument("-o", "--output", help="Output .npy for the embedding (default: print summary).")

    s = _add(sub, "sample", "Write a bundled demo EEG epoch to a .npy you can compress.")
    s.add_argument("--idx", type=int, default=0, help="Which bundled epoch (default 0).")
    s.add_argument("-o", "--output", default="epoch.npy", help="Output .npy path.")

    _add(sub, "info", "Show available tiers and their measured scores.")

    dl = _add(sub, "download", "Download the trained model bundle into ~/.neurozip.")
    dl.add_argument("--url", help="Override the model bundle URL.")
    dl.add_argument("--dest", help="Install dir (default ~/.neurozip).")

    for name in ("serve", "ui", "server"):
        sv = _add(sub, name, "Launch the local upload UI (real compression).")
        sv.add_argument("--port", type=int, default=7878)
        sv.add_argument("--host", default="127.0.0.1")
        sv.add_argument("--no-browser", action="store_true")
    return p


def _require_ml():
    """Exit with a friendly message (not a traceback) if the ml extra is absent."""
    import importlib.util
    missing = [m for m in ("numpy", "torch")
               if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit("This command needs the ML extra and the models:\n"
                 "  pip install 'neurozip[ml]'\n  neurozip download")


def _load_input(args):
    import numpy as np
    from . import runtime as rt
    if getattr(args, "sample", None) is not None:
        s = rt.load_samples()
        if s is None:
            sys.exit("No bundled samples found. Run `neurozip download` or pass a .npy input.")
        eeg, concepts = s
        if args.sample < 0 or args.sample >= len(eeg):
            sys.exit(f"--sample must be 0..{len(eeg)-1}")
        return eeg[args.sample], concepts[args.sample]
    if not args.input:
        sys.exit("Provide an input .npy or --sample IDX.")
    return np.load(args.input), None


def _cmd_compress(args) -> int:
    _require_ml()
    import numpy as np
    from . import runtime as rt
    eeg, concept = _load_input(args)
    out = args.output or ((args.input or f"sample{args.sample}") + ".nz")
    stats = rt.compress(eeg, tier=args.tier, variant=args.variant)
    rt.save_nz(out, stats)
    tag = f" [{concept}]" if concept else ""
    print(f"[neurozip] compressed{tag} with {args.variant} ({args.tier}): "
          f"bpp={stats['bpp']:.4f}  ratio={stats['ratio_vs_fp16']:.1f}x vs fp16  "
          f"latent={stats['latent_shape']}  ->  {out}")
    return 0


def _cmd_decompress(args) -> int:
    _require_ml()
    import numpy as np
    from . import runtime as rt
    arc = rt.load_nz(args.input)
    recon = rt.decompress(arc["latents"], tier=arc["tier"], variant=arc["variant"])
    out = args.output or (args.input + ".recon.npy")
    np.save(out, recon)
    print(f"[neurozip] decompressed {args.input} ({arc['variant']} {arc['tier']}) "
          f"-> {out}  shape={recon.shape}")
    return 0


def _cmd_embed(args) -> int:
    _require_ml()
    import numpy as np
    from . import runtime as rt
    eeg, concept = _load_input(args)
    z = rt.embed(eeg)
    if args.output:
        np.save(args.output, z)
        print(f"[neurozip] embedding {z.shape} -> {args.output}")
    else:
        v = z.reshape(-1)
        print(f"[neurozip] embedding dim={v.shape[0]} norm={float((v**2).sum()**0.5):.3f} "
              f"first5={np.round(v[:5], 4).tolist()}")
    return 0


def _cmd_sample(args) -> int:
    _require_ml()
    import numpy as np
    from . import runtime as rt
    s = rt.load_samples()
    if s is None:
        sys.exit("No bundled samples found. Run `neurozip download`.")
    eeg, concepts = s
    if args.idx < 0 or args.idx >= len(eeg):
        sys.exit(f"--idx must be 0..{len(eeg)-1}")
    np.save(args.output, eeg[args.idx])
    print(f"[neurozip] wrote demo epoch #{args.idx} ({concepts[args.idx]}) -> {args.output}")
    return 0


def _cmd_info(args) -> int:
    from . import runtime as rt
    sc = rt.scores()
    print(f"checkpoints: {rt.checkpoints_dir()}")
    if not sc:
        print("no scores.json found.")
        return 0
    print(f"{'model':22s} {'bpp':>7} {'ratio':>7} {'mse':>7} {'img@1':>6} {'txt@1':>6}")
    for name in sorted(sc):
        s = sc[name]
        print(f"{name:22s} {s.get('bpp',0):7.4f} "
              f"{s.get('compression_ratio_vs_fp16',0):6.1f}x {s.get('mse',0):7.4f} "
              f"{s.get('retrieval_image',{}).get('top1',0):6.2f} "
              f"{s.get('retrieval_text',{}).get('top1',0):6.2f}")
    return 0


def _cmd_download(args) -> int:
    from . import runtime as rt
    rt.download_models(url=args.url, dest=args.dest)
    return 0


def _cmd_serve(args) -> int:
    from . import server
    return server.run(host=args.host, port=args.port, open_browser=not args.no_browser)


ML_COMMANDS = {"compress", "decompress", "embed", "sample", "serve", "ui", "server"}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    # Auto-setup: build the ML env and re-exec into it on first inference use,
    # then make sure the trained models are present (both show a status).
    from . import bootstrap
    bootstrap.reexec_if_needed(args.command, ML_COMMANDS)
    if args.command in ML_COMMANDS:
        from . import runtime as rt
        rt.ensure_models()
    dispatch = {
        "compress": _cmd_compress, "decompress": _cmd_decompress,
        "embed": _cmd_embed, "sample": _cmd_sample, "info": _cmd_info,
        "download": _cmd_download, "serve": _cmd_serve, "ui": _cmd_serve,
        "server": _cmd_serve,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
