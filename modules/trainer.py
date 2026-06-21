"""
Trainer for DCAF radiology report generation model.

Implements training loop with:
- Multi-component loss (L_con, L_match, L_cls, L_gen)
- Validation and test evaluation
- Best model checkpointing
- Distributed training support
"""

import os
import copy
from abc import abstractmethod

import torch
import torch.distributed as dist
import numpy as np
from numpy import inf

from .metrics_clinical import CheXbertMetrics
from .optims import LinearWarmupCosineLRScheduler


class BaseTrainer:
    """Base trainer with optimizer setup and training loop."""

    def __init__(
        self,
        model,
        criterion_cls,
        base_probs,
        metric_ftns,
        args,
        device,
        is_main_process,
    ):
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process

        # CheXbert clinical metrics
        self.chexbert_metrics = CheXbertMetrics(
            './checkpoints/stanford/chexbert/chexbert.pth',
            args.batch_size,
            device,
        )

        self.criterion_cls = criterion_cls
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns

        # ── Optimizer setup ──────────────────────────────────────────────────
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print(f"Number of trainable parameters: {num_parameters:,}")

        optim_params = [
            {"params": p_wd, "weight_decay": float(self.args.weight_decay)},
            {"params": p_non_wd, "weight_decay": 0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, 0.999),
        )

        self.epochs = self.args.epochs
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_best = 0
        self.log_best = {}
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            result = self._train_epoch(epoch)
            # Synchronise all processes before evaluation (DDP only)
            if self.args.distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()
            result = self._eval_epoch(result)

            # Save log
            log = {'epoch': epoch}
            log.update(result)

            # Record best and save checkpoint
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = copy.deepcopy(log)
                    best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                    # Save model state dict (handle DDP wrapper)
                    model_to_save = (
                        self.model.module
                        if hasattr(self.model, 'module')
                        else self.model
                    )
                    torch.save(model_to_save.state_dict(), best_path)
                    print(f"Saving current best to {best_path}")

            # Print log
            for key, value in log.items():
                print(f'\t{str(key):15s}: {value}')

        if self.is_main_process:
            print(f'Best results w.r.t {self.mnt_metric}:')
            for key, value in self.log_best.items():
                print(f'\t{str(key):15s}: {value}')


class DCAFTrainer(BaseTrainer):
    """
    Trainer for DCAF model.

    Handles the multi-component loss:
        L = L_con^D + lambda_match * gamma + lambda_cls * L_cls + lambda_gen * L_gen
    """

    def __init__(
        self,
        model,
        criterion_cls,
        base_probs,
        metric_ftns,
        args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        is_main_process,
    ):
        super().__init__(
            model, criterion_cls, base_probs, metric_ftns,
            args, device, is_main_process
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            self.args.epochs,
            self.args.min_lr,
            self.args.init_lr,
            decay_rate=None,
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )

    def _train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.

        Returns:
            dict with 'train_loss' and component losses
        """
        self.model.train()
        train_loss = 0.0
        train_loss_gen = 0.0
        train_loss_cls = 0.0
        train_loss_con = 0.0
        train_loss_match = 0.0

        for batch_idx, (
            images, captions, cls_labels, retrieved_text_feats, retrieved_text_labels
        ) in enumerate(self.train_dataloader):

            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            retrieved_text_feats = retrieved_text_feats.to(self.device)
            retrieved_text_labels = retrieved_text_labels.to(self.device)

            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)

            loss, loss_gen, loss_cls, loss_con, loss_match = self.model(
                images,
                captions,
                cls_labels,
                retrieved_text_feats,
                retrieved_text_labels,
                self.criterion_cls,
                self.base_probs,
            )

            if batch_idx % 10 == 0:
                print(
                    f"[{batch_idx}/{len(self.train_dataloader)}] "
                    f"loss: {loss.item():.4f} | "
                    f"gen: {loss_gen.item():.4f} | "
                    f"cls: {loss_cls.item():.4f} | "
                    f"con: {loss_con.item():.4f} | "
                    f"match: {loss_match.item():.4f}"
                )

            train_loss += loss.item()
            train_loss_gen += loss_gen.item()
            train_loss_cls += loss_cls.item()
            train_loss_con += loss_con.item()
            train_loss_match += loss_match.item()

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        n = len(self.train_dataloader)
        log = {
            'train_loss': train_loss / n,
            'train_loss_gen': train_loss_gen / n,
            'train_loss_cls': train_loss_cls / n,
            'train_loss_con': train_loss_con / n,
            'train_loss_match': train_loss_match / n,
        }
        return log

    def _eval_epoch(self, log: dict) -> dict:
        """
        Evaluate on validation and test sets.

        Returns:
            Updated log dict with val/test metrics
        """
        model_eval = (
            self.model.module if hasattr(self.model, 'module') else self.model
        )
        model_eval.eval()

        # ── Validation ───────────────────────────────────────────────────────
        logits_list = []
        counts_list = []

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (
                images, captions, cls_labels, retrieved_text_feats, retrieved_text_labels
            ) in enumerate(self.val_dataloader):

                images = images.to(self.device)
                cls_labels = cls_labels.to(self.device)
                retrieved_text_feats = retrieved_text_feats.to(self.device)

                ground_truths = captions
                reports, cls_preds, cls_preds_logits = model_eval.generate(
                    images,
                    retrieved_text_feats,
                    self.base_probs,
                    sample=False,
                    num_beams=self.args.beam_size,
                    max_length=self.args.gen_max_len,
                    min_length=self.args.gen_min_len,
                )

                # Logit adjustment tracking
                cls_labels_bin = (cls_labels == 1).float()
                logit = cls_preds_logits * cls_labels_bin
                logits_list.append(logit.cpu().numpy())
                counts_list.append(cls_labels_bin.cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            # Update base_probs with validation statistics
            logits_arr = np.concatenate(logits_list, axis=0)
            counts_arr = np.concatenate(counts_list, axis=0)
            logits_sum = np.sum(logits_arr, 0)
            counts_sum = np.sum(counts_arr, 0)
            # Avoid division by zero
            counts_sum = np.maximum(counts_sum, 1e-8)
            logits_mean = logits_sum / counts_sum
            logits_mean /= np.maximum(np.max(logits_mean), 1e-8)
            # Append 4 auxiliary disease priors
            logits_mean = np.append(logits_mean, [1, 1, 1, 1])
            self.base_probs = logits_mean  # update class distribution

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_ce.items()})

        # ── Test ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (
                images, captions, cls_labels, retrieved_text_feats, retrieved_text_labels
            ) in enumerate(self.test_dataloader):

                images = images.to(self.device)
                retrieved_text_feats = retrieved_text_feats.to(self.device)

                ground_truths = captions
                reports, _, _ = model_eval.generate(
                    images,
                    retrieved_text_feats,
                    self.base_probs,
                    sample=False,
                    num_beams=self.args.beam_size,
                    max_length=self.args.gen_max_len,
                    min_length=self.args.gen_min_len,
                )

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)},
            )
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})

        return log
