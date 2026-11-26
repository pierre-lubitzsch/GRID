"""Sentence-T5 item embeddings, the content encoder of TIGER (Rajput et al. 2023).

Encodes each item's text (title, brand, categories, price: the dataset's `text` feature, the
fields TIGER uses) with sentence-transformers/sentence-t5-base, reproducing that library's
pipeline without depending on it: T5 encoder -> mean over real tokens -> the model's Dense
projection (768 -> 768, no bias) -> L2 normalization. Output: a float tensor (num_items, 768),
row i = item id i, the same layout as embeddings/<ds>_merged_predictions_tensor_latest.pt.

Usage (from GRID/): python -m scripts.sentence_t5_embeddings --dataset beauty \
    --out embeddings/beauty_st5_tensor.pt
"""

import argparse
import glob
import os

import torch


def read_items(items_dir: str):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    files = sorted(glob.glob(os.path.join(items_dir, "*.tfrecord*")))
    if not files:
        raise SystemExit(f"no item shards under {items_dir}")
    out = {}
    for r in tf.data.TFRecordDataset(files, compression_type="GZIP"):
        ex = tf.train.Example()
        ex.ParseFromString(r.numpy())
        f = ex.features.feature
        out[int(f["id"].int64_list.value[0])] = f["text"].bytes_list.value[0].decode("utf-8")
    ids = sorted(out)
    if ids != list(range(len(ids))):
        raise SystemExit(f"item ids are not 0..N-1 ({len(ids)} items, max {ids[-1]})")
    return [out[i] for i in ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="src/data/amazon_data")
    ap.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, T5EncoderModel

    texts = read_items(os.path.join(a.data_root, a.dataset, "items"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = T5EncoderModel.from_pretrained(a.model).to(dev).eval()
    dense_w = load_file(hf_hub_download(a.model, "2_Dense/model.safetensors"))["linear.weight"].to(dev)
    print(f"[st5] {a.dataset}: {len(texts)} items, dense {tuple(dense_w.shape)}, device {dev}", flush=True)
    print(f"[st5] example: {texts[0]!r}", flush=True)
    outs, n_trunc = [], 0
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            b = tok(texts[i:i + a.batch], padding=True, truncation=True, max_length=a.max_len,
                    return_tensors="pt").to(dev)
            n_trunc += int((b["attention_mask"].sum(1) >= a.max_len).sum())
            h = enc(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            outs.append(torch.nn.functional.normalize(pooled @ dense_w.T, dim=-1).cpu())
    emb = torch.cat(outs)
    print(f"[st5] embeddings {tuple(emb.shape)}, {n_trunc} texts hit max_len {a.max_len}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(emb, a.out)
    print(f"[st5] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
