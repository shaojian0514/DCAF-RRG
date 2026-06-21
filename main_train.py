"""
Main training script for DCAF: Disease-Consistent Visual-Text Alignment
and Fusion for Reliable Radiology Report Generation.

Usage:
    # Single GPU
    python main_train.py --dataset_name mimic_cxr --image_dir data/mimic_cxr/images/ \
        --ann_path data/mimic_cxr/mimic_annotation_promptmrg.json

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 main_train.py --distributed True \
        --dataset_name mimic_cxr ...
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import numpy as np

from modules.metrics import compute_scores
from modules.trainer import DCAFTrainer
from models.dcaf import dcaf_model
from dataset import create_dataset, create_sampler, create_loader
from modules import utils
from transformers import BertTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(
        description='DCAF: Disease-Consistent Visual-Text Alignment and Fusion'
    )

    # ── Data settings ────────────────────────────────────────────────────────
    parser.add_argument('--image_dir', type=str,
                        default='data/mimic_cxr/images/',
                        help='Path to image directory.')
    parser.add_argument('--ann_path', type=str,
                        default='data/mimic_cxr/mimic_annotation_promptmrg.json',
                        help='Path to annotation JSON file.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size.')

    # ── Dataset settings ─────────────────────────────────────────────────────
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr',
                        choices=['iu_xray', 'mimic_cxr'],
                        help='Dataset to use.')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Word frequency cutoff threshold.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU.')

    # ── Model settings ───────────────────────────────────────────────────────
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint.')
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

    # ── Training settings ────────────────────────────────────────────────────
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/dcaf',
                        help='Directory to save checkpoints.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1',
                        help='Metric to monitor for best model selection.')

    # ── Optimization ─────────────────────────────────────────────────────────
    parser.add_argument('--init_lr', type=float, default=5e-4,
                        help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate.')
    parser.add_argument('--warmup_lr', type=float, default=5e-6,
                        help='Warmup starting learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps.')

    # ── Loss weights ─────────────────────────────────────────────────────────
    parser.add_argument('--cls_weight', type=float, default=4.0,
                        help='Weight for auxiliary classification loss.')
    parser.add_argument('--lambda_match', type=float, default=1.0,
                        help='Weight for disease-matching constraint loss.')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='Weight for focal disease classification loss.')
    parser.add_argument('--lambda_gen', type=float, default=1.0,
                        help='Weight for generation loss.')

    # ── Retrieval settings ───────────────────────────────────────────────────
    parser.add_argument('--clip_k', type=int, default=18,
                        help='Number of retrieved reports (top-k).')
    parser.add_argument('--clip_feat_path', type=str,
                        default='data/mimic_cxr/clip_text_features.json',
                        help='Path to pre-computed CLIP text features.')
    parser.add_argument('--clip_label_path', type=str,
                        default='data/mimic_cxr/clip_text_labels.json',
                        help='Path to disease labels for retrieved reports.')

    # ── Distributed training ─────────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=9233,
                        help='Random seed.')
    parser.add_argument('--distributed', default=True, type=bool,
                        help='Use distributed training.')
    parser.add_argument('--dist_url', default='env://',
                        help='URL for distributed training setup.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu).')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize distributed training
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix random seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_dataset(
        'generation_%s' % args.dataset_name, tokenizer, args
    )
    print(f'Training samples:   {len(train_dataset):,}')
    print(f'Validation samples: {len(val_dataset):,}')
    print(f'Test samples:       {len(test_dataset):,}')

    # Load class frequency priors for logit adjustment
    with open('./data/mimic_cxr/base_probs.json', 'r') as f:
        base_probs = json.load(f)
    base_probs = np.array(base_probs) / np.max(base_probs)
    base_probs = np.append(base_probs, [1, 1, 1, 1])  # 4 auxiliary diseases

    # ── Distributed samplers ─────────────────────────────────────────────────
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset],
            [True, False, False],
            num_tasks,
            global_rank,
        )
        samplers = [samplers[0], None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size] * 3,
        num_workers=[args.num_workers] * 3,
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Build prompt template for prompt_length calculation
    labels_temp = ['[BLA]'] * 14  # 14 disease categories
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

    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.load_pretrained}")
        print(f"Missing keys: {msg.missing_keys}")

    # Auxiliary classification criterion
    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = DCAFTrainer(
        model=model,
        criterion_cls=criterion_cls,
        base_probs=base_probs,
        metric_ftns=metrics,
        args=args,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        device=device,
        is_main_process=utils.is_main_process,
    )
    trainer.train()


if __name__ == '__main__':
    main()
