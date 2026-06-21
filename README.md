# DCAF: Disease-Consistent Visual-Text Alignment and Fusion for Reliable Radiology Report Generation

<p align="center">
  <a href="https://miccai.org/"><img src="https://img.shields.io/badge/MICCAI-2026-blue" alt="MICCAI 2026"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/cite-BibTeX-green" alt="Citation"></a>
  <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-orange" alt="PyTorch">
</p>

> **[MICCAI 2026]** Disease-Consistent Visual-Text Alignment and Fusion for Reliable Radiology Report Generation

---

## Abstract

Automatic medical report generation (MRG) has significant clinical value, as it can alleviate the heavy workload of report writing for radiologists. Recent studies adopt retrieval-based strategies to inject external diagnostic knowledge into report generation. However, reliable MRG remains challenging because existing methods often retrieve diagnostically inconsistent reports and lack mechanisms for selective visual-text integration.

In this work, we propose **DCAF**, a **D**isease-**C**onsistent Visual-Text **A**lignment and **F**usion framework for reliable radiology report generation. DCAF incorporates:

- A **Disease-Consistent Retrieval Enhancer (DRE)** that integrates disease priors via multi-positive contrastive learning and a disease-matching constraint to improve diagnostic consistency during retrieval.
- An **Optimal Transport-Enhanced Cross-Modal Fusion (OTCF)** module that selectively integrates clinically relevant textual knowledge to mimic the selective diagnostic reasoning process of radiologists.
- An auxiliary **Disease-aware Feature Extraction (DFE)** mechanism that encodes predicted diagnostic hypotheses into visual representations, enabling hypothesis-driven visual reasoning.

Extensive experiments on IU X-Ray and MIMIC-CXR demonstrate state-of-the-art performance in both language generation quality and clinical efficacy.

---

## Framework Overview

```
Input X-ray Image
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          DCAF Framework                              │
│                                                                      │
│  ┌─────────────────────┐                                             │
│  │   ResNet-101        │  Visual Encoder (2048-d → 256-d projection) │
│  └──────────┬──────────┘                                             │
│             │  F_c (visual features)                                 │
│             ├──────────────────────────────────────────┐             │
│             │                                          │             │
│  ┌──────────▼──────────┐    ┌────────────────────┐    │             │
│  │        DRE          │    │       OTCF          │    │             │
│  │ Disease-Consistent  │    │ Optimal Transport   │    │             │
│  │ Retrieval Enhancer  │    │ Cross-Modal Fusion  │    │             │
│  │                     │    │                     │    │             │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │             │
│  │  │ DMCL          │  │    │  │ MPSA          │  │    │             │
│  │  │ Multi-positive│  │    │  │ Multi-Prompt  │  │    │             │
│  │  │ Contrastive   │  │    │  │ Sinkhorn      │  │    │             │
│  │  │ Learning      │  │    │  │ Attention     │  │    │             │
│  │  └───────────────┘  │    │  └───────────────┘  │    │             │
│  │  ┌───────────────┐  │    └────────┬───────────┘    │             │
│  │  │ DCR           │  │             │ F_fused         │             │
│  │  │ Disease-match │  │    ┌────────▼───────────┐    │             │
│  │  │ Constrained   │  │    │       DFE           │◄───┘             │
│  │  │ Retrieval     │  │    │ Disease-aware       │                  │
│  │  └───────────────┘  │    │ Feature Extraction  │                  │
│  └─────────────────────┘    │ F_d = ŷ·Φ + F_c    │                  │
│                             └────────┬───────────┘                  │
│                                      │                               │
│                             ┌────────▼───────────┐                  │
│                             │   BERT-LM Decoder   │                  │
│                             │  (bert-base-uncased) │                  │
│                             └────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
Generated Radiology Report
```

---

## Key Components

### 1. DRE — Disease-Consistent Retrieval Enhancer

**DMCL** (Disease-prior Multi-positive Contrastive Learning, Eq. 1):

- Projects image and text features into a disease-aware embedding space `(B, num_diseases, embed_dim)` via a learnable disease-aware projection layer.
- Within a training batch, reports sharing **identical disease labels** with the query image are treated as positive samples; mismatched reports are negatives.
- Optimises a multi-positive contrastive objective to explicitly model disease-level consistency.

**DCR** (Disease-matching Constrained Retrieval, Eq. 2):

- Enforces that retrieved reports have disease distributions consistent with the query image via a BCE-based matching loss.
- Filters out disease-inconsistent retrievals at training time, improving the clinical reliability of reference reports.

### 2. OTCF — Optimal Transport-Enhanced Cross-Modal Fusion

**MPSA** (Multi-Prompt Sinkhorn Attention, Eq. 3):

- Replaces standard cross-attention with Sinkhorn-Knopp optimal transport to selectively integrate clinically relevant textual knowledge.
- Cost matrix: `C = 1 − cosine_sim(Q_norm, K_norm)` (non-negative, bounded in [0, 2]).
- Sinkhorn algorithm runs in **log-domain** for numerical stability, iterating to convergence with regularisation `ε = 0.05`.
- Transport plan `T` is used to compute the fused feature: `output = T @ V`.

### 3. DFE — Disease-aware Feature Extraction

- Per-disease binary Softmax classification head over 14 CheXpert disease categories (Eq. 4).
- **Focal Loss** (Eq. 5) to handle severe label imbalance.
- Disease-aware feature: `F_d = ŷ · Φ + F_c` (Eq. 4), where `ŷ` are predicted disease probabilities and `Φ` is a learnable disease embedding matrix.
- Logit adjustment with class-frequency priors to correct for training distribution bias.

### Overall Training Loss (Eq. 7)

```
L = L_con^D + λ_match · γ + λ_cls · L_cls + λ_gen · L_gen
```

| Term                | Description                              |
| ------------------- | ---------------------------------------- |
| `L_con^D`         | DMCL multi-positive contrastive loss     |
| `λ_match · γ`  | DCR disease-matching constraint loss     |
| `λ_cls · L_cls` | Focal classification loss (DFE)          |
| `λ_gen · L_gen` | Language generation loss (cross-entropy) |

---

## Repository Structure

```
DCAF-RRG/
├── models/
│   ├── dcaf.py              # Main DCAF model (integrates DRE + OTCF + DFE)
│   ├── dre.py               # Disease-Consistent Retrieval Enhancer
│   ├── otcf.py              # Optimal Transport Cross-Modal Fusion (MPSA)
│   ├── dfe.py               # Disease-aware Feature Extraction Classifier
│   ├── med.py               # BERT-LM text decoder (adapted from BLIP)
│   └── resnet.py            # ResNet-101 visual encoder
├── modules/
│   ├── trainer.py           # DCAFTrainer (training + evaluation loop)
│   ├── chexbert.py          # CheXbert clinical metric computation
│   ├── metrics_clinical.py  # CE metrics (Precision / Recall / F1)
│   ├── metrics.py           # NLG metrics (BLEU / METEOR / ROUGE)
│   ├── optims.py            # LinearWarmupCosine LR scheduler
│   ├── tokenizers.py        # Tokenizer utilities
│   └── utils.py             # Distributed training utilities
├── dataset/
│   ├── medical_dataset.py   # DCAFTrainDataset / DCAFEvalDataset
│   ├── __init__.py          # Dataset factory (create_dataset, create_loader)
│   └── utils.py             # Image transforms
├── configs/
│   └── bert_config.json     # BERT decoder configuration
├── data/
│   ├── mimic_cxr/
│   │   ├── base_probs.json              # Class-frequency priors for logit adjustment
│   │   └── mimic_annotation_dcaf.json  # Annotation file (to be prepared)
│   └── iu_xray/
│       └── annotation.json             # Annotation file (to be prepared)
├── checkpoints/
│   └── stanford/chexbert/   # Place chexbert.pth here
├── pycocoevalcap/           # NLG evaluation toolkit
├── main_train.py            # Training entry point
├── main_test.py             # Testing entry point
├── build_retrieval_queue.py # Pre-compute CLIP text features for retrieval
├── train_mimic_cxr.sh       # Training script (MIMIC-CXR)
├── test_mimic_cxr.sh        # Testing script (MIMIC-CXR)
├── test_iu_xray.sh          # Testing script (IU X-Ray)
└── requirements.txt
```

---

## Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3 (recommended)
- Java (required for METEOR evaluation)

### Installation

```bash
# 1. Clone the repository
git clone <repo_url>
cd DCAF-RRG

# 2. Create conda environment
conda create -n dcaf python=3.8
conda activate dcaf

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

---

## Data Preparation

### MIMIC-CXR

1. Apply for access at [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
2. Download images and reports.
3. Prepare the annotation JSON file in the following format:

```json
{
  "train": [
    {
      "id": "study_id",
      "image_path": ["p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"],
      "report": "The cardiomediastinal silhouette is normal...",
      "split": "train",
      "label": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    }
  ],
  "val": [...],
  "test": [...]
}
```

The `label` field contains 14 CheXpert disease labels:

- `0`: Blank / Negative / Uncertain (treated as negative)
- `1`: Positive (confirmed presence)

4. Place the annotation file at `data/mimic_cxr/mimic_annotation_dcaf.json`.
5. Place images at `data/mimic_cxr/images/`.

**Dataset statistics** (from paper):

| Split | Samples |
| ----- | ------- |
| Train | 270,790 |
| Val   | 2,130   |
| Test  | 3,858   |

### IU X-Ray

Download from [OpenI](https://openi.nlm.nih.gov/faq) and prepare annotation in the same format.
Place at `data/iu_xray/annotation.json` and images at `data/iu_xray/images/`.

---

## CheXbert Checkpoint

Download the CheXbert model checkpoint from [Stanford Box](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) and place it at:

```
checkpoints/stanford/chexbert/chexbert.pth
```

---

## Build Retrieval Queue

Before training, pre-compute CLIP text features for all training reports:

```bash
python build_retrieval_queue.py \
    --ann_path data/mimic_cxr/mimic_annotation_dcaf.json \
    --output_path data/mimic_cxr/retrieval_queue.pkl \
    --clip_model ViT-B/32 \
    --batch_size 256
```

This creates a retrieval queue file containing:

- CLIP text features for all training reports: `(N_train, 512)`
- Disease labels for all training reports: `(N_train, 14)`

The dataset loader uses this queue to retrieve the top-k most similar reports at training/inference time.

---

## Training

### Single GPU

```bash
bash train_mimic_cxr.sh
```

Or manually:

```bash
python main_train.py \
    --dataset_name mimic_cxr \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/mimic_annotation_dcaf.json \
    --save_dir results/dcaf_mimic_cxr \
    --embed_dim 256 \
    --num_heads 8 \
    --sinkhorn_eps 0.05 \
    --temperature 0.07 \
    --clip_k 18 \
    --batch_size 32 \
    --epochs 10 \
    --init_lr 5e-4 \
    --weight_decay 0.01 \
    --cls_weight 4.0 \
    --lambda_match 1.0 \
    --lambda_cls 1.0 \
    --lambda_gen 1.0 \
    --beam_size 3 \
    --distributed False \
    --monitor_metric ce_f1
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 main_train.py \
    --dataset_name mimic_cxr \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/mimic_annotation_dcaf.json \
    --save_dir results/dcaf_mimic_cxr \
    --distributed True \
    [... other args ...]
```

### Key Hyperparameters

| Parameter        | Value | Description                                      |
| ---------------- | ----- | ------------------------------------------------ |
| `embed_dim`    | 256   | Shared embedding dimension                       |
| `num_heads`    | 8     | Number of attention heads in OTCF                |
| `sinkhorn_eps` | 0.05  | Sinkhorn regularisation ε (runs to convergence) |
| `temperature`  | 0.07  | Contrastive learning temperature τ              |
| `clip_k`       | 18    | Number of retrieved reference reports            |
| `batch_size`   | 32    | Per-GPU batch size                               |
| `epochs`       | 10    | Training epochs                                  |
| `init_lr`      | 5e-4  | Initial learning rate (AdamW)                    |
| `weight_decay` | 0.01  | Weight decay                                     |
| `cls_weight`   | 4.0   | Auxiliary classification loss weight             |
| `lambda_match` | 1.0   | DCR constraint loss weight λ_match              |
| `lambda_cls`   | 1.0   | Focal classification loss weight λ_cls          |
| `lambda_gen`   | 1.0   | Generation loss weight λ_gen                    |

---

## Testing

### MIMIC-CXR

```bash
bash test_mimic_cxr.sh
```

Or manually:

```bash
python main_test.py \
    --dataset_name mimic_cxr \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/mimic_annotation_dcaf.json \
    --load_pretrained results/dcaf_mimic_cxr/model_best.pth \
    --save_dir results/dcaf_mimic_cxr \
    --clip_k 18 \
    --beam_size 3 \
    --save_reports
```

### IU X-Ray

```bash
bash test_iu_xray.sh
```

---

## Results

### MIMIC-CXR (Table 1 from paper)

| Method                | BLEU-1          | BLEU-4          | METEOR          | ROUGE-L         | CE-P            | CE-R            | CE-F1           |
| --------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| R2Gen                 | 0.353           | 0.103           | 0.142           | 0.277           | 0.333           | 0.273           | 0.276           |
| RGRG                  | 0.363           | 0.107           | 0.149           | 0.284           | 0.447           | 0.298           | 0.341           |
| PromptMRG             | 0.398           | 0.128           | 0.163           | 0.306           | 0.480           | 0.380           | 0.415           |
| **DCAF (Ours)** | **0.412** | **0.137** | **0.171** | **0.318** | **0.501** | **0.402** | **0.438** |

### IU X-Ray (Table 1 from paper)

| Method                | BLEU-1          | BLEU-4          | METEOR          | ROUGE-L         |
| --------------------- | --------------- | --------------- | --------------- | --------------- |
| R2Gen                 | 0.470           | 0.165           | 0.187           | 0.371           |
| PromptMRG             | 0.483           | 0.175           | 0.196           | 0.382           |
| **DCAF (Ours)** | **0.498** | **0.183** | **0.204** | **0.394** |

*CE metrics: CheXbert Precision / Recall / F1 over 14 CheXpert disease categories.*

---

## Evaluation Metrics

### NLG Metrics (Natural Language Generation)

- **BLEU-1/4**: n-gram precision against reference reports
- **METEOR**: Unigram F-score with stemming and synonymy matching
- **ROUGE-L**: Longest common subsequence F-score

### CE Metrics (Clinical Efficacy)

- **CheXbert Precision / Recall / F1**: Computed using the CheXbert labeler on 14 CheXpert disease categories
- Measures the clinical accuracy of generated reports independent of surface-form wording

---

## Acknowledgements

This codebase builds upon:

- [PromptMRG](https://github.com/jhb86253817/PromptMRG): Prompt-guided multi-label recognition for radiology report generation
- [BLIP](https://github.com/salesforce/BLIP): Bootstrapping Language-Image Pre-training
- [CheXbert](https://github.com/stanfordmlgroup/CheXbert): Automated chest X-ray report labeling
- [CLIP](https://github.com/openai/CLIP): Contrastive Language-Image Pre-training
