"""
fable_cli.py -- compress / decompress 4D fMRI NIfTI files with the fable codec.

    python fable_cli.py compress   sub_bold.nii.gz -o out.nzf   --ckpt checkpoints/fable_nd3.pt
    python fable_cli.py decompress out.nzf -o recon.nii.gz      --ckpt checkpoints/fable_nd3.pt
    python fable_cli.py info out.nzf

The .nzf is self-contained (given the per-subject codec bundle): a small CLI
header (original 4D shape, affine, crop bounds) wraps the entropy-coded token
stream + static side-info. `decompress` rebuilds a full-size NIfTI in original
scanner units. The codec is per-subject (its stored mu/mask calibrate one
subject's space); train one per subject with fable_train.py + fable_finalize.py.
"""
from __future__ import annotations
import argparse, struct
import numpy as np
import torch
import nibabel as nib
import fable_data, fable_io
from fable_codec import VQCodec

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CMAGIC = b"NZC1"


def _load_bundle(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    if "prior" not in ck:
        raise SystemExit(f"{path} has no temporal prior; run: python fable_finalize.py --ckpt {path}")
    codec = VQCodec(**ck["config"]).to(DEV); codec.load_state_dict(ck["model"]); codec.eval()
    p = ck["prior"]
    codec._prior = {"marginal": p["marginal"].astype(np.float64),
                    "transition": p["transition"].astype(np.float64), "K": int(p["K"])}
    mask = ck["meta"]["mask"]; mu = ck["meta"]["mu"]; gstd = float(ck["meta"]["sd"][0])
    return codec, mask, mu, gstd


def cmd_compress(args):
    codec, mask, mu, gstd = _load_bundle(args.ckpt)
    img = nib.load(args.input)
    affine = img.affine.astype(np.float64)
    bold = img.get_fdata(dtype=np.float32)                       # (X,Y,Z,T)
    orig_shape = np.array(bold.shape, np.int32)
    cx, cy, cz = fable_data.CROP
    crop = bold[cx, cy, cz, :]                                   # (D,H,W,T)
    vols = np.moveaxis(crop, -1, 0)                              # (T,D,H,W)
    z = np.zeros_like(vols); z[:, mask] = (vols[:, mask] - mu) / gstd
    blob, info = fable_io.compress(z, codec, codec._prior, mu, gstd, mask, device=DEV)
    cbounds = np.array([cx.start, cx.stop, cy.start, cy.stop, cz.start, cz.stop], np.int32)
    header = CMAGIC + affine.tobytes() + orig_shape.tobytes() + cbounds.tobytes() + struct.pack("<I", len(blob))
    with open(args.output, "wb") as f:
        f.write(header + blob)
    total = len(header) + len(blob)
    print(f"compressed {args.input} -> {args.output}")
    print(f"  {info['n_vols']} volumes, {total/1024:.1f} KB ({total/info['n_vols']:.0f} B/vol)")


def _read_container(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == CMAGIC, "not an NZC1 file"
    off = 4
    affine = np.frombuffer(buf[off:off+128], np.float64).reshape(4, 4); off += 128
    orig_shape = np.frombuffer(buf[off:off+16], np.int32); off += 16
    cbounds = np.frombuffer(buf[off:off+24], np.int32); off += 24
    (n_blob,) = struct.unpack_from("<I", buf, off); off += 4
    return affine, orig_shape, cbounds, buf[off:off+n_blob]


def cmd_decompress(args):
    codec, mask, mu, gstd = _load_bundle(args.ckpt)
    affine, orig_shape, cb, blob = _read_container(args.input)
    rec, mu2, gstd2, mask2 = fable_io.decompress(blob, codec, device=DEV)
    T = rec.shape[0]
    full = np.zeros((*orig_shape[:3], T), np.float32)
    xhat = np.zeros_like(rec); xhat[:, mask2] = rec[:, mask2] * gstd2 + mu2
    full[cb[0]:cb[1], cb[2]:cb[3], cb[4]:cb[5], :] = np.moveaxis(xhat, 0, -1)
    nib.save(nib.Nifti1Image(full, affine), args.output)
    print(f"decompressed {args.input} -> {args.output}  shape {full.shape}")


def cmd_info(args):
    affine, orig_shape, cb, blob = _read_container(args.input)
    off = len(fable_io.MAGIC)
    (T, gd, gh, gw, K, D, H, W, gstd, n_mask, n_mu) = struct.unpack_from("<IHHHHHHHfII", blob, off)
    import os
    sz = os.path.getsize(args.input)
    print(f"{args.input}")
    print(f"  original 4D shape : {tuple(int(x) for x in orig_shape)}")
    print(f"  volumes           : {T}")
    print(f"  latent grid       : ({gd},{gh},{gw}) = {gd*gh*gw} tokens/vol, K={K}")
    print(f"  file size         : {sz/1024:.1f} KB ({sz/T:.0f} B/vol)")


def main():
    ap = argparse.ArgumentParser(description="NeuroZip fMRI-fable codec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress"); c.add_argument("input"); c.add_argument("-o", "--output", required=True)
    c.add_argument("--ckpt", default="checkpoints/fable_nd3.pt"); c.set_defaults(fn=cmd_compress)
    dc = sub.add_parser("decompress"); dc.add_argument("input"); dc.add_argument("-o", "--output", required=True)
    dc.add_argument("--ckpt", default="checkpoints/fable_nd3.pt"); dc.set_defaults(fn=cmd_decompress)
    i = sub.add_parser("info"); i.add_argument("input"); i.set_defaults(fn=cmd_info)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
