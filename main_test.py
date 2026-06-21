"""
Main test/inference script for DCAF.

Usage:
    # Test on MIMIC-CXR
    python main_test.py --dataset_name mimic_cxr \
        --image_dir data/mimic_cxr/images/ \
        --ann_path data/mimic_cxr/mimic_annotation_promptmrg.json \
        --load_pretrained results/dcaf/model_best.pth

    # Test on IU X-Ray (using MIMIC-CXR pretrained model)
    python main_test.py --dataset_name iu_xray \
        --image_dir data/iu_xray/images/ \
        --ann_path data/iu_xray/annotation.json \
        --load_pretrained results/dcaf/model_best.pth
"""

import os
import json
import argparse

import torch
import numpy as np

from modules.metrics import compute_scores
from modules.metrics_clinical import CheXbertMetrics
from models.dcaf import dcaf_model
from dataset import create_dataset_test, create_loader
from modules import utils
from transformers import BertTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(
        description='DCAF Test/Inference Script'
    )

    # ── Data settings ────────────────────────────────────────────────────────
    parser.add_argument('--image_dir', type=str,
                        default='data/iu_xray/images/',
                        help='Path to image directory.')
    parser.add_argument('--ann_path', type=str,
                        default='data/iu_xray/annotation.json',
                        help='Path to annotation JSON file.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size.')

    # ── Dataset settings ─────────────────────────────────────────────────────
    parser.add_argument('--dataset_name', type=str, default='iu_xray',
                        choices=['iu_xray', 'mimic_cxr'],
                        help='Dataset to evaluate on.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')

    # ── Model settings ───────────────────────────────────────────────────────
    parser.add_argument('--load_pretrained', type=str, required=True,
                        help='Path to pretrained DCAF checkpoint.')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Shared embedding dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.05,
                        help='Sinkhorn regularisation epsilon for OTCF (runs to convergence).')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning.')

    # ── Generation settings ──────────────────────────────────────────────────
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size for beam search.')
    parser.add_argument('--gen_max_len', type=int, default=150,
                        help='Maximum generation length.')
    parser.add_argument('--gen_min_len', type=int, default=100,
                        help='Minimum generation length.')

    # ── Retrieval settings ───────────────────────────────────────────────────
    parser.add_argument('--clip_k', type=int, default=18,
                        help='Number of retrieved reports (top-k).')
    parser.add_argument('--clip_feat_path', type=str,
                        default='data/mimic_cxr/clip_text_features.json',
                        help='Path to pre-computed CLIP text features.')
    parser.add_argument('--clip_label_path', type=str,
                        default='data/mimic_cxr/clip_text_labels.json',
                        help='Path to disease labels for retrieved reports.')

    # ── Output settings ──────────────────────────────────────────────────────
    parser.add_argument('--save_dir', type=str, default='results/dcaf',
                        help='Directory to save results.')
    parser.add_argument('--save_reports', action='store_true',
                        help='Save generated reports to file.')

    # ── Other settings ───────────────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=9233,
                        help='Random seed.')
    parser.add_argument('--distributed', default=False, type=bool,
                        help='Use distributed evaluation.')
    parser.add_argument('--dist_url', default='env://',
                        help='URL for distributed setup.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use.')
    parser.add_argument('--cls_weight', type=float, default=4.0,
                        help='Auxiliary classification loss weight (unused in test).')
    parser.add_argument('--lambda_match', type=float, default=1.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_gen', type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Creating test dataset...")
    test_dataset = create_dataset_test(
        'generation_%s' % args.dataset_name, tokenizer, args
    )
    print(f'Test samples: {len(test_dataset):,}')

    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[args.batch_size],
        num_workers=[args.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    # ── Load class frequency priors ──────────────────────────────────────────
    with open('./data/mimic_cxr/base_probs.json', 'r') as f:
        base_probs = json.load(f)
    base_probs = np.array(base_probs) / np.max(base_probs)
    base_probs = np.append(base_probs, [1, 1, 1, 1])

    # ── Model ─────────────────────────────────────────────────────────────────
    labels_temp = ['[BLA]'] * 14
    prompt_temp = ' '.join(labels_temp) + ' '

    model = dcaf_model(
        args,
        tokenizer,
        image_size=args.image_size,
        prompt=prompt_temp,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        sinkhorn_eps=args.sinkhorn_eps,
        temperature=args.temperature,
        lambda_match=args.lambda_match,
        lambda_cls=args.lambda_cls,
        lambda_gen=args.lambda_gen,
    )

    state_dict = torch.load(args.load_pretrained, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.load_pretrained}")

    model = model.to(device)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference...")
    chexbert_metrics = CheXbertMetrics(
        './checkpoints/stanford/chexbert/chexbert.pth',
        args.batch_size,
        device,
    )

    test_gts, test_res = [], []
    with torch.no_grad():
        for batch_idx, (
            images, captions, cls_labels, retrieved_text_feats, retrieved_text_labels
        ) in enumerate(test_loader):

            images = images.to(device)
            retrieved_text_feats = retrieved_text_feats.to(device)

            reports, _, _ = model.generate(
                images,
                retrieved_text_feats,
                base_probs,
                sample=False,
                num_beams=args.beam_size,
                max_length=args.gen_max_len,
                min_length=args.gen_min_len,
            )

            test_res.extend(reports)
            test_gts.extend(captions)

            if batch_idx % 10 == 0:
                print(f"  [{batch_idx}/{len(test_loader)}]")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")
    nlg_metrics = compute_scores(
        {i: [gt] for i, gt in enumerate(test_gts)},
        {i: [re] for i, re in enumerate(test_res)},
    )
    ce_metrics = chexbert_metrics.compute(test_gts, test_res)

    print("\n" + "=" * 60)
    print("NLG Metrics:")
    for k, v in nlg_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("\nClinical Efficacy Metrics:")
    for k, v in ce_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save_reports:
        os.makedirs(args.save_dir, exist_ok=True)
        results = {
            'nlg_metrics': nlg_metrics,
            'ce_metrics': ce_metrics,
            'generated_reports': test_res,
            'ground_truth_reports': test_gts,
        }
        save_path = os.path.join(args.save_dir, f'test_results_{args.dataset_name}.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
