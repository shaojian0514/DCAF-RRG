"""
Build Retrieval Queue for DCAF.

This script pre-computes CLIP text features for all training reports and
builds the retrieval index used during training and inference.

For each sample in the dataset, it:
1. Encodes all training reports using CLIP text encoder
2. Computes cosine similarity between each sample and all training reports
3. Stores top-k indices for fast retrieval during training

Output files:
    data/mimic_cxr/clip_text_features.json  - CLIP text features for all reports
    data/mimic_cxr/clip_text_labels.json    - Disease labels for all reports
    data/mimic_cxr/mimic_annotation_dcaf.json - Annotation with clip_indices

Usage:
    python build_retrieval_queue.py \
        --ann_path data/mimic_cxr/mimic_annotation_promptmrg.json \
        --output_dir data/mimic_cxr/ \
        --top_k 30 \
        --batch_size 256

Note:
    This script requires a CLIP model. We use the pre-trained CLIP
    (ViT-B/32) from OpenAI. The text features are computed once and
    cached for efficient training.

    If you already have clip_text_features.json from PromptMRG, you can
    reuse it directly. Just ensure clip_text_labels.json is also present.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build retrieval queue for DCAF'
    )
    parser.add_argument('--ann_path', type=str,
                        default='data/mimic_cxr/mimic_annotation_promptmrg.json',
                        help='Path to annotation JSON file.')
    parser.add_argument('--output_dir', type=str,
                        default='data/mimic_cxr/',
                        help='Output directory for feature files.')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Number of top-k reports to retrieve per sample.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for CLIP encoding.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use.')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model variant.')
    return parser.parse_args()


def encode_texts_with_clip(texts, model, tokenizer, device, batch_size=256):
    """
    Encode a list of texts using CLIP text encoder.

    Args:
        texts: list of strings
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: torch device
        batch_size: encoding batch size
    Returns:
        (N, D) numpy array of L2-normalized text features
    """
    all_features = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding texts'):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(batch, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def build_retrieval_index(
    query_features: np.ndarray,
    bank_features: np.ndarray,
    top_k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Build retrieval index by computing cosine similarity.

    Args:
        query_features: (N, D) query features
        bank_features: (M, D) bank features
        top_k: number of top retrievals
        batch_size: batch size for similarity computation
    Returns:
        (N, top_k) indices into bank_features
    """
    N = query_features.shape[0]
    all_indices = []

    query_tensor = torch.from_numpy(query_features).float()
    bank_tensor = torch.from_numpy(bank_features).float()

    for i in tqdm(range(0, N, batch_size), desc='Building index'):
        q_batch = query_tensor[i:i + batch_size]  # (B, D)
        # Cosine similarity: (B, M)
        sim = torch.matmul(q_batch, bank_tensor.T)
        # Top-k indices
        _, indices = sim.topk(top_k, dim=-1)
        all_indices.append(indices.numpy())

    return np.concatenate(all_indices, axis=0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load annotation ───────────────────────────────────────────────────────
    print(f"Loading annotation from {args.ann_path}...")
    with open(args.ann_path, 'r') as f:
        annotation = json.load(f)

    # Collect all training reports and their disease labels
    train_ann = annotation['train']
    train_reports = [ann['report'] for ann in train_ann]
    train_labels = [ann['labels'] for ann in train_ann]

    print(f"Total training reports: {len(train_reports):,}")

    # ── Load CLIP model ───────────────────────────────────────────────────────
    try:
        import clip
        print(f"Loading CLIP model: {args.clip_model}...")
        clip_model, _ = clip.load(args.clip_model, device=device)
        clip_model.eval()
        clip_tokenizer = clip.tokenize
    except ImportError:
        print("ERROR: CLIP not installed. Please install: pip install git+https://github.com/openai/CLIP.git")
        print("\nAlternatively, if you already have clip_text_features.json from PromptMRG,")
        print("you can skip this script and just create clip_text_labels.json manually.")
        return

    # ── Encode all training reports ───────────────────────────────────────────
    print("Encoding training reports with CLIP...")
    text_features = encode_texts_with_clip(
        train_reports, clip_model, clip_tokenizer, device, args.batch_size
    )
    print(f"Text features shape: {text_features.shape}")

    # ── Save text features ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    feat_path = os.path.join(args.output_dir, 'clip_text_features.json')
    print(f"Saving text features to {feat_path}...")
    with open(feat_path, 'w') as f:
        json.dump(text_features.tolist(), f)

    # ── Save disease labels for all reports ───────────────────────────────────
    label_path = os.path.join(args.output_dir, 'clip_text_labels.json')
    print(f"Saving text labels to {label_path}...")
    with open(label_path, 'w') as f:
        json.dump(train_labels, f)

    # ── Build retrieval index for all splits ──────────────────────────────────
    print(f"\nBuilding retrieval index (top-{args.top_k})...")

    # Process each split
    updated_annotation = {}
    for split in annotation.keys():
        split_ann = annotation[split]
        split_reports = [ann['report'] for ann in split_ann]

        print(f"\nProcessing {split} split ({len(split_reports):,} samples)...")

        # Encode split reports
        split_features = encode_texts_with_clip(
            split_reports, clip_model, clip_tokenizer, device, args.batch_size
        )

        # Build retrieval index: for each sample, find top-k from training bank
        indices = build_retrieval_index(
            split_features, text_features, args.top_k
        )

        # Update annotation with clip_indices
        updated_split = []
        for i, ann in enumerate(split_ann):
            ann_copy = dict(ann)
            ann_copy['clip_indices'] = indices[i].tolist()
            updated_split.append(ann_copy)
        updated_annotation[split] = updated_split

    # ── Save updated annotation ───────────────────────────────────────────────
    output_ann_path = os.path.join(args.output_dir, 'mimic_annotation_dcaf.json')
    print(f"\nSaving updated annotation to {output_ann_path}...")
    with open(output_ann_path, 'w') as f:
        json.dump(updated_annotation, f)

    print("\nDone! Retrieval queue built successfully.")
    print(f"  Text features: {feat_path}")
    print(f"  Text labels:   {label_path}")
    print(f"  Annotation:    {output_ann_path}")


if __name__ == '__main__':
    main()
