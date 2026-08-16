"""E0b — inference-time field ablation: what does the 84M table actually buy?

Zero training cost. Loads the frozen pretrained dense checkpoint
(`cache/checkpoints/dcnv2_vanilla.pt`, the same Arm-A model whose test AUC is 0.7775 in
docs/20260814-2111) and, one field at a time, zeroes that field's embedding
table so its contribution to the concatenated 360-d input becomes the zero
vector. The drop in pooled test AUC is that field's *dependence* score.

Interpretation boundary (registered before running, do not weaken later):
  - This measures DEPENDENCE of the trained model on a field, NOT the field's
    necessity under retraining (the model never got a chance to compensate).
    Therefore it OVER-estimates importance: a field whose ablation costs ~0
    provably cannot be a capacity bottleneck, while a field whose ablation
    costs a lot is only a *candidate*.
  - Used for exactly one decision: whether widening the embedding table can
    possibly pay off. A near-zero drop on video_id/author_id/music_id (99.96%
    of embedding params) closes the "embedding capacity gap" hypothesis
    without spending a single training run.

Also reports the OOV/seen split for video_id: pooled test AUC restricted to
rows whose video_id was never updated during training vs rows that were.
Cross-subgroup AUC levels are not directly comparable (different base rates);
only the *within-subgroup* ablation deltas are used as evidence.
"""

import json
import os
import sys

import numpy as np
import torch
from torcheval.metrics import BinaryAUROC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fields  # noqa: E402
from dataset import GpuBatches, Split  # noqa: E402
from model import DCNv2  # noqa: E402

CACHE_DIR = "cache"
VANILLA_PATH = f"{CACHE_DIR}/checkpoints/dcnv2_vanilla.pt"
OUT_DIR = f"{CACHE_DIR}/embedding_capacity"
OUT_JSON = f"{OUT_DIR}/field_ablation.json"
BATCH_SIZE = 10000


def _logits(model, src):
    outs, labels, vids = [], [], []
    for batch in src:
        with torch.inference_mode():
            model(batch)
        outs.append(batch["logit"].float())
        labels.append(batch["is_click"].float())
        vids.append(batch["video_id"])
    return torch.cat(outs), torch.cat(labels), torch.cat(vids)


def _auc(logit, label, mask=None):
    m = BinaryAUROC()
    if mask is None:
        m.update(logit, label)
    else:
        m.update(logit[mask], label[mask])
    return float(m.compute())


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_JSON):
        raise SystemExit(f"{OUT_JSON} exists; evidence bundles are immutable.")

    model = DCNv2().to(device)
    model.load_state_dict(torch.load(VANILLA_PATH, map_location=device))
    model.requires_grad_(False)
    model.eval()

    train, _valid, test = Split("all")
    seen_video = np.zeros(fields.video["video_id"], dtype=bool)
    seen_video[np.unique(train["video_id"].to_numpy())] = True
    seen_video_t = torch.from_numpy(seen_video).to(device)
    del train

    test_src = GpuBatches(test, BATCH_SIZE, device, shuffle=False)
    del test

    logit, label, vid = _logits(model, test_src)
    base_auc = _auc(logit, label)
    vid_seen_mask = seen_video_t[vid.long()]
    base_seen = _auc(logit, label, vid_seen_mask)
    base_oov = _auc(logit, label, ~vid_seen_mask)
    n_seen = int(vid_seen_mask.sum())
    n_oov = int((~vid_seen_mask).sum())
    print(f"[base] test AUC = {base_auc:.6f}  "
          f"(video_id seen n={n_seen:,} AUC={base_seen:.6f} | "
          f"OOV n={n_oov:,} AUC={base_oov:.6f})")

    vocab = fields.user | fields.video
    results = []
    for field in vocab:
        table = model.sparse.tables[field]
        saved = table.weight.detach().clone()
        table.weight.detach().zero_()
        a_logit, a_label, _ = _logits(model, test_src)
        row = {
            "field": field,
            "vocab": vocab[field],
            "params": vocab[field] * 10,
            "test_auc_ablated": _auc(a_logit, a_label),
            "test_auc_ablated_video_seen": _auc(a_logit, a_label, vid_seen_mask),
            "test_auc_ablated_video_oov": _auc(a_logit, a_label, ~vid_seen_mask),
        }
        row["delta_auc"] = row["test_auc_ablated"] - base_auc
        row["delta_auc_video_seen"] = (
            row["test_auc_ablated_video_seen"] - base_seen)
        row["delta_auc_video_oov"] = (
            row["test_auc_ablated_video_oov"] - base_oov)
        table.weight.detach().copy_(saved)
        results.append(row)
        print(f"  ablate {field:<24} test AUC={row['test_auc_ablated']:.6f} "
              f"Δ={row['delta_auc']:+.6f}")

    # combined ablation of the three high-cardinality id fields (99.96% params)
    saved = {}
    for field in ("video_id", "author_id", "music_id"):
        t = model.sparse.tables[field]
        saved[field] = t.weight.detach().clone()
        t.weight.detach().zero_()
    c_logit, c_label, _ = _logits(model, test_src)
    combo = {
        "fields": ["video_id", "author_id", "music_id"],
        "params": 83984250,
        "test_auc_ablated": _auc(c_logit, c_label),
        "test_auc_ablated_video_seen": _auc(c_logit, c_label, vid_seen_mask),
        "test_auc_ablated_video_oov": _auc(c_logit, c_label, ~vid_seen_mask),
    }
    combo["delta_auc"] = combo["test_auc_ablated"] - base_auc
    for field, w in saved.items():
        model.sparse.tables[field].weight.detach().copy_(w)
    print(f"\n[combo] ablate big-3 ids (83,984,250 params): "
          f"test AUC={combo['test_auc_ablated']:.6f} "
          f"Δ={combo['delta_auc']:+.6f}")

    results.sort(key=lambda r: r["delta_auc"])
    bundle = {
        "provenance": {
            "script": "scripts/diagnose_field_ablation.py",
            "checkpoint": VANILLA_PATH,
            "device": device,
            "batch_size": BATCH_SIZE,
            "method": "zero the field's embedding table at inference time",
            "boundary": "dependence of the trained model, not necessity under "
                        "retraining; over-estimates importance",
        },
        "base_test_auc": base_auc,
        "base_test_auc_video_seen": base_seen,
        "base_test_auc_video_oov": base_oov,
        "test_rows_video_seen": n_seen,
        "test_rows_video_oov": n_oov,
        "per_field_ablation": results,
        "big_three_combined_ablation": combo,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
