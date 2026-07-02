"""
fable_finalize.py -- turn a trained checkpoint into a self-contained codec bundle.

Training saves the network weights + per-subject calibration (mu, sd, mask). To
compress standalone we also need the temporal token prior P(cur | prev), which is
estimated once on the training tokens. This step computes it and writes it back
into the checkpoint so `fable_cli.py` needs nothing but the bundle.

    python fable_finalize.py --ckpt checkpoints/fable_nd3.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import fable_data, fable_io
from fable_codec import VQCodec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    codec = VQCodec(**ck["config"]).to(DEV); codec.load_state_dict(ck["model"]); codec.eval()
    tr = fable_data.load()["train"]
    with torch.no_grad():
        codes = torch.cat([codec.encode(torch.from_numpy(tr[i:i+128]).unsqueeze(1).to(DEV))
                           for i in range(0, len(tr), 128)]).reshape(len(tr), -1).cpu().numpy().astype(np.int32)
    prior = fable_io.fit_prior(codes, ck["config"]["K"])
    ck["prior"] = {"marginal": prior["marginal"].astype(np.float32),
                   "transition": prior["transition"].astype(np.float32),
                   "K": prior["K"]}
    torch.save(ck, args.ckpt)
    print(f"[finalize] wrote temporal prior into {args.ckpt} "
          f"(marginal {prior['marginal'].shape}, transition {prior['transition'].shape})")


if __name__ == "__main__":
    main()
