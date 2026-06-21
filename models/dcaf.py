"""
DCAF: Disease-Consistent Visual-Text Alignment and Fusion
for Reliable Radiology Report Generation

Main model integrating:
1. Disease-Consistent Retrieval Enhancer (DRE)
   - Multi-positive Contrastive Learning (DMCL)
   - Disease-matching Constrained Retrieval (DCR)
2. Optimal Transport-Enhanced Cross-Modal Fusion (OTCF)
3. Disease Classifier for Disease-aware Feature Extraction (DFE)
4. Text Decoder (BERT-based autoregressive generation)

Overall loss (Eq. 7):
    L = L_con^D + lambda_match * gamma + lambda_cls * L_cls + lambda_gen * L_gen
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from models.med import BertConfig, BertLMHeadModel
from models.resnet import blip_resnet
from models.dre import DiseaseConsistentRetrievalEnhancer
from models.otcf import OptimalTransportCrossModalFusion
from models.dfe import DiseaseClassifier


# 14 CheXpert disease categories
CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

NUM_DISEASES = len(CONDITIONS)  # 14

# Diagnosis score tokens (same as PromptMRG)
SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']


class DCAF(nn.Module):
    """
    DCAF: Disease-Consistent Visual-Text Alignment and Fusion framework.

    Architecture:
        - Visual Encoder: ResNet-101 (pretrained)
        - DRE: Disease-Consistent Retrieval Enhancer
        - OTCF: Optimal Transport-Enhanced Cross-Modal Fusion
        - DFE: Disease Classifier for Disease-aware Feature Extraction
        - Text Decoder: BERT-LM (autoregressive report generation)

    Args:
        args: argument namespace with model hyperparameters
        tokenizer: BertTokenizer
        image_size: input image resolution (default 224)
        prompt: prompt prefix string for generation
        num_diseases: number of disease categories (default 14)
        embed_dim: shared embedding dimension (default 256)
        num_heads: number of attention heads (default 8)
        sinkhorn_eps: Sinkhorn regularisation epsilon for OTCF (default 0.05)
        temperature: contrastive learning temperature (default 0.07)
        lambda_match: weight for DCR constraint loss
        lambda_cls: weight for disease classification loss
        lambda_gen: weight for generation loss
    """

    def __init__(
        self,
        args,
        tokenizer,
        image_size: int = 224,
        prompt: str = '',
        num_diseases: int = NUM_DISEASES,
        embed_dim: int = 256,
        num_heads: int = 8,
        sinkhorn_eps: float = 0.05,
        # backward-compat alias: sinkhorn_iters is ignored (Sinkhorn now runs to convergence)
        sinkhorn_iters: int = 3,
        temperature: float = 0.07,
        lambda_match: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_gen: float = 1.0,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        # Loss weights
        self.lambda_match = lambda_match
        self.lambda_cls = lambda_cls
        self.lambda_gen = lambda_gen

        # ── Visual Encoder (ResNet-101) ──────────────────────────────────────
        self.visual_encoder = blip_resnet(args)
        vision_width = 2048  # ResNet-101 output channels

        # CLIP text feature dimension (from pre-computed retrieval queue)
        text_feat_dim = 512

        # ── Disease-Consistent Retrieval Enhancer (DRE) ──────────────────────
        self.dre = DiseaseConsistentRetrievalEnhancer(
            image_feat_dim=vision_width,
            text_feat_dim=text_feat_dim,
            num_diseases=num_diseases,
            embed_dim=embed_dim,
            temperature=temperature,
        )

        # ── Visual feature projection to embed_dim (for OTCF input) ─────────
        self.visual_patch_proj = nn.Linear(vision_width, embed_dim)

        # ── Optimal Transport-Enhanced Cross-Modal Fusion (OTCF) ─────────────
        self.otcf = OptimalTransportCrossModalFusion(
            visual_dim=embed_dim,
            text_dim=text_feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            sinkhorn_eps=sinkhorn_eps,
        )

        # ── Disease Classifier for Disease-aware Feature Extraction (DFE) ────
        self.dfe = DiseaseClassifier(
            num_diseases=num_diseases,
            embed_dim=embed_dim,
        )

        # ── Text Decoder (BERT-LM) ───────────────────────────────────────────
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        # Decoder cross-attends to (B, 2*num_diseases, embed_dim) encoder states
        decoder_config.encoder_width = embed_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained(
            'bert-base-uncased', config=decoder_config
        )
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # Prompt settings
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # ── Auxiliary classification head (logit-adjustment, same as PromptMRG) ──
        # Input: avg visual feat (vision_width) + retrieved text summary (embed_dim)
        self.cls_head = nn.Linear(vision_width + embed_dim, num_diseases * 4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass (training)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,
        caption: list,
        cls_labels: torch.Tensor,
        retrieved_text_feats: torch.Tensor,
        retrieved_text_labels: torch.Tensor,
        criterion_cls,
        base_probs,
    ) -> tuple:
        """
        Training forward pass.

        Args:
            image: (B, C, H, W) input chest X-ray images
            caption: list of B report strings (with diagnosis prompt prefix)
            cls_labels: (B, num_diseases) ground-truth disease labels {0,1,2,3}
                        (0=blank, 1=positive, 2=negative, 3=uncertain)
            retrieved_text_feats: (B, k, text_feat_dim) top-k retrieved text features
            retrieved_text_labels: (B, num_diseases) disease labels of the most
                                   relevant retrieved report (for DMCL)
            criterion_cls: classification criterion (CrossEntropyLoss)
            base_probs: (num_diseases+4,) class frequency priors for logit adjustment
        Returns:
            total_loss, loss_gen, loss_cls, loss_con, loss_match
        """
        B = image.size(0)

        # ── 1. Visual Encoding ───────────────────────────────────────────────
        # patch_feats: (B, L, vision_width), avg_feats: (B, vision_width)
        patch_feats, avg_feats = self.visual_encoder(image)

        # ── 2. DRE: Disease-aware Image Embedding ────────────────────────────
        # image_disease_feats: (B, num_diseases, embed_dim)
        image_disease_feats = self.dre.encode_image(avg_feats)

        # ── 3. DRE: Multi-positive Contrastive Loss (DMCL) ──────────────────
        top1_text_feat = retrieved_text_feats[:, 0, :]          # (B, text_feat_dim)
        text_disease_feats = self.dre.encode_text(top1_text_feat)  # (B, d, e)

        # Binary labels: 1=positive, others=0
        binary_cls_labels = (cls_labels == 1).long()   # (B, num_diseases)
        binary_ret_labels = (retrieved_text_labels == 1).long()

        loss_con = self.dre.compute_contrastive_loss(
            image_disease_feats,
            text_disease_feats,
            binary_cls_labels,
            binary_ret_labels,
        )

        # ── 4. DRE: Disease-matching Constraint (DCR) ────────────────────────
        k = retrieved_text_feats.size(1)
        ret_feats_flat = retrieved_text_feats.view(B * k, -1)          # (B*k, text_feat_dim)
        ret_disease_feats = self.dre.encode_text(ret_feats_flat)        # (B*k, d, e)
        ret_disease_feats_mean = ret_disease_feats.mean(dim=1)          # (B*k, e)
        ret_disease_feats_mean = ret_disease_feats_mean.view(B, k, -1)  # (B, k, e)

        loss_match = self.dre.compute_matching_constraint(
            ret_disease_feats_mean,
            binary_cls_labels.float(),
        )

        # ── 5. OTCF: Cross-Modal Fusion ──────────────────────────────────────
        Fv = self.visual_patch_proj(patch_feats)   # (B, L, embed_dim)
        Fg = top1_text_feat.unsqueeze(1)            # (B, 1, text_feat_dim)
        Fc_patch = self.otcf(Fv, Fg)               # (B, L, embed_dim)

        # Pool to disease-level: add avg-pooled fused feat to disease-aware image feat
        Fc_avg = Fc_patch.mean(dim=1)                              # (B, embed_dim)
        Fc = image_disease_feats + Fc_avg.unsqueeze(1)             # (B, num_diseases, embed_dim)

        # ── 6. DFE: Disease-aware Feature Extraction ─────────────────────────
        Fd, y_hat, loss_cls = self.dfe(Fc, binary_cls_labels)     # (B, d, e), (B, d, 2), scalar

        # ── 7. Auxiliary Classification Head (logit adjustment) ──────────────
        retrieved_summary = retrieved_text_feats.mean(dim=1)       # (B, text_feat_dim)
        retrieved_proj = self.dre.text_proj(retrieved_summary).mean(dim=1)  # (B, embed_dim)
        cls_input = torch.cat([avg_feats, retrieved_proj], dim=1)  # (B, vision_width+embed_dim)
        cls_preds = self.cls_head(cls_input)                        # (B, num_diseases*4)
        cls_preds = cls_preds.view(-1, 4, self.num_diseases)        # (B, 4, num_diseases)

        # Logit adjustment (non-in-place to preserve gradients)
        base_probs_tensor = torch.from_numpy(
            np.array(base_probs[:self.num_diseases])
        ).float().to(image.device)
        log_prior = torch.log(base_probs_tensor).view(1, 1, -1)    # (1, 1, num_diseases)
        cls_preds = cls_preds.clone()
        cls_preds[:, 1:2, :] = cls_preds[:, 1:2, :] + log_prior   # add only to positive class

        # Auxiliary CE loss for prompt generation
        loss_cls_aux = criterion_cls(cls_preds, cls_labels)

        # ── 8. Text Decoder: Report Generation ──────────────────────────────
        encoder_hidden = torch.cat([Fc, Fd], dim=1)                # (B, 2*d, embed_dim)
        encoder_atts = torch.ones(
            encoder_hidden.size()[:-1], dtype=torch.long
        ).to(image.device)

        text = self.tokenizer(
            caption,
            padding='longest',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        loss_gen = decoder_output.loss

        # Combined loss (Eq. 7)
        total_loss = (
            loss_con
            + self.lambda_match * loss_match
            + self.lambda_cls * loss_cls
            + self.lambda_gen * loss_gen
            + self.args.cls_weight * loss_cls_aux
        )

        return total_loss, loss_gen, loss_cls, loss_con, loss_match

    # ─────────────────────────────────────────────────────────────────────────
    # Generation (inference)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        retrieved_text_feats: torch.Tensor,
        base_probs,
        sample: bool = False,
        num_beams: int = 3,
        max_length: int = 150,
        min_length: int = 100,
        repetition_penalty: float = 1.0,
    ) -> tuple:
        """
        Inference: generate radiology report for given image.

        Args:
            image: (B, C, H, W) input chest X-ray images
            retrieved_text_feats: (B, k, text_feat_dim) retrieved text features
            base_probs: array-like class frequency priors for logit adjustment
            sample: whether to use sampling (False = beam search)
            num_beams: beam size for beam search
            max_length: maximum generation length
            min_length: minimum generation length
            repetition_penalty: repetition penalty for generation
        Returns:
            captions: list of B generated report strings
            cls_preds_argmax: list of B disease prediction lists
            cls_preds_logits: (B, num_diseases) positive class probabilities
        """
        B = image.size(0)

        # ── 1. Visual Encoding ───────────────────────────────────────────────
        patch_feats, avg_feats = self.visual_encoder(image)

        # ── 2. DRE: Disease-aware Image Embedding ────────────────────────────
        image_disease_feats = self.dre.encode_image(avg_feats)

        # ── 3. OTCF: Cross-Modal Fusion ──────────────────────────────────────
        Fv = self.visual_patch_proj(patch_feats)       # (B, L, embed_dim)
        top1_text_feat = retrieved_text_feats[:, 0, :] # (B, text_feat_dim)
        Fg = top1_text_feat.unsqueeze(1)               # (B, 1, text_feat_dim)
        Fc_patch = self.otcf(Fv, Fg)                   # (B, L, embed_dim)
        Fc_avg = Fc_patch.mean(dim=1)                  # (B, embed_dim)
        Fc = image_disease_feats + Fc_avg.unsqueeze(1) # (B, num_diseases, embed_dim)

        # ── 4. DFE: Disease-aware Feature Extraction ─────────────────────────
        Fd, y_hat, _ = self.dfe(Fc)

        # ── 5. Auxiliary Classification for Prompt Generation ────────────────
        retrieved_summary = retrieved_text_feats.mean(dim=1)       # (B, text_feat_dim)
        retrieved_proj = self.dre.text_proj(retrieved_summary).mean(dim=1)  # (B, embed_dim)
        cls_input = torch.cat([avg_feats, retrieved_proj], dim=1)
        cls_preds_raw = self.cls_head(cls_input)
        cls_preds_raw = cls_preds_raw.view(-1, 4, self.num_diseases)  # (B, 4, num_diseases)

        # Logit adjustment
        base_probs_arr = np.array(base_probs[:self.num_diseases], dtype=np.float32)
        base_probs_tensor = torch.from_numpy(base_probs_arr).to(image.device)
        log_prior = torch.log(base_probs_tensor).view(1, 1, -1)
        cls_preds_raw = cls_preds_raw.clone()
        cls_preds_raw[:, 1:2, :] = cls_preds_raw[:, 1:2, :] + log_prior

        cls_preds_softmax = F.softmax(cls_preds_raw, dim=1)
        cls_preds_logits = cls_preds_softmax[:, 1, :]              # (B, num_diseases)
        cls_preds_argmax = torch.argmax(cls_preds_softmax, dim=1).cpu().numpy().tolist()

        # Build diagnosis prompts
        prompts = []
        for j in range(B):
            prompt = ' '.join([SCORES[c] for c in cls_preds_argmax[j]]) + ' '
            prompts.append(prompt)

        # ── 6. Text Decoder: Beam Search Generation ──────────────────────────
        encoder_hidden = torch.cat([Fc, Fd], dim=1)   # (B, 2*d, embed_dim)
        encoder_atts = torch.ones(
            encoder_hidden.size()[:-1], dtype=torch.long
        ).to(image.device)

        if not sample:
            encoder_hidden = encoder_hidden.repeat_interleave(num_beams, dim=0)
            encoder_atts = encoder_atts.repeat_interleave(num_beams, dim=0)

        model_kwargs = {
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": encoder_atts,
        }

        text = self.tokenizer(prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        attn_masks = attn_masks[:, :-1]

        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            min_length=min_length,
            max_new_tokens=max_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            attention_mask=attn_masks,
            **model_kwargs,
        )

        captions = []
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prompts[i]):])

        return captions, cls_preds_argmax, cls_preds_logits


def dcaf_model(args, tokenizer, **kwargs) -> DCAF:
    """Factory function to create DCAF model."""
    model = DCAF(args, tokenizer, **kwargs)
    return model
