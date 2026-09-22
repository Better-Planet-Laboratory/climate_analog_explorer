
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

BIO_NAMES = [f"BIO{i}" for i in range(1, 20)]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_stage2_head(path, embed_dim=384, device="cpu"):
    # Import from your existing Stage-2 script so architecture stays identical.
    from train_search_head_stage2 import SearchHead
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    head = SearchHead(
        embed_dim,
        args.get("search_hidden_dim", 256),
        args.get("search_dim", 64),
    ).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    return head, ck

def encode_pooled(model, encoder, climate, pixel_mask, lat):
    patch_valid = model.get_patch_validity(pixel_mask, 0.50)
    tokens, H, W = encoder(climate, lat)
    return model.pool_tokens(tokens, patch_valid)

def coarse_fields_numpy(climate, mask, out_size=8, min_valid_fraction=0.50):
    """
    climate: torch [B,19,64,64], already historically standardized
    mask:    torch [B,64,64] or [B,1,64,64]
    Returns:
      coarse [B,19,8,8] float32
      valid  [B,8,8] bool
    Uses masked average pooling, keeping support separate from climate values.
    """
    if mask.ndim == 3:
        mask = mask[:, None]
    mask = mask.float()
    weighted = climate * mask
    num = F.adaptive_avg_pool2d(weighted, (out_size, out_size))
    den = F.adaptive_avg_pool2d(mask, (out_size, out_size))
    coarse = num / den.clamp_min(1e-8)
    valid = den[:, 0] >= float(min_valid_fraction)
    coarse = coarse.masked_fill(~valid[:, None], 0.0)
    return coarse.cpu().numpy().astype(np.float32), valid.cpu().numpy()

def exact_climate_distance(query_climate, query_valid, ref_climate, ref_valid,
                           min_shared_cells=8, chunk=4096):
    """
    Corrected shared-valid climate distance used for final reranking.
    Shapes:
      query_climate [19,8,8]
      query_valid   [8,8]
      ref_climate   [N,19,8,8]
      ref_valid     [N,8,8]
    Returns float32 [N], inf where insufficient shared support.
    """
    q = np.asarray(query_climate, np.float32)
    qv = np.asarray(query_valid, bool)
    n = len(ref_climate)
    out = np.full(n, np.inf, dtype=np.float32)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        r = np.asarray(ref_climate[s:e], np.float32)
        rv = np.asarray(ref_valid[s:e], bool)
        shared = rv & qv[None]
        counts = shared.sum(axis=(1,2))
        ok = counts >= min_shared_cells
        if not ok.any():
            continue

        # Mean squared difference over channels AND shared cells.
        diff2 = (r - q[None]) ** 2
        w = shared[:, None].astype(np.float32)
        denom = counts.astype(np.float32) * q.shape[0]
        mse = (diff2 * w).sum(axis=(1,2,3)) / np.maximum(denom, 1.0)
        d = np.sqrt(mse, dtype=np.float32)
        d[~ok] = np.inf
        out[s:e] = d
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(np.asarray(lat2))
    lon2 = np.radians(np.asarray(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def period_file_name(period):
    return f"{period}_search.h5"

def read_manifest(path):
    with open(path) as f:
        return json.load(f)
