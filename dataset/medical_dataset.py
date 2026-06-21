"""
Dataset classes for DCAF radiology report generation.

Supports:
- MIMIC-CXR (train/val/test splits)
- IU X-Ray (full dataset, evaluated with MIMIC-CXR pretrained model)

Each sample returns:
- image: transformed chest X-ray image tensor
- caption: report string (with diagnosis prompt prefix for training)
- cls_labels: (num_diseases,) disease label tensor {0=blank,1=pos,2=neg,3=unc}
- retrieved_text_feats: (k, text_feat_dim) top-k retrieved text features
- retrieved_text_labels: (num_diseases,) disease labels of top-1 retrieved report
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption

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

# Diagnosis score tokens
SCORES = ['[BLA]', '[POS]', '[NEG]', '[UNC]']


class DCAFTrainDataset(Dataset):
    """
    Training dataset for DCAF.

    Annotation JSON format (same as PromptMRG):
    {
        "train": [
            {
                "image_path": ["path/to/image.jpg"],
                "report": "report text",
                "labels": [0, 1, 2, ...],          # 14 disease labels
                "clip_indices": [idx1, idx2, ...],  # indices into text feature bank
                "clip_labels": [[0,1,...], ...]     # disease labels of retrieved reports
            },
            ...
        ]
    }

    Args:
        transform: image transform pipeline
        image_root: root directory for images
        ann_root: path to annotation JSON file
        tokenizer: BertTokenizer
        max_words: maximum report length in words
        dataset: 'mimic_cxr' or 'iu_xray'
        args: argument namespace (clip_k, etc.)
    """

    def __init__(
        self,
        transform,
        image_root: str,
        ann_root: str,
        tokenizer,
        max_words: int = 100,
        dataset: str = 'mimic_cxr',
        args=None,
    ):
        self.annotation = json.load(open(ann_root, 'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.dataset = dataset
        self.args = args

        # Load pre-computed CLIP text features for retrieval
        # feat_path can be overridden via args.clip_feat_path
        feat_path = getattr(args, 'clip_feat_path', None) or \
            './data/mimic_cxr/clip_text_features.json'
        with open(feat_path, 'r') as f:
            self.clip_features = np.array(json.load(f))

        # Load disease labels for all reports in the bank (for DMCL)
        label_path = getattr(args, 'clip_label_path', None) or \
            './data/mimic_cxr/clip_text_labels.json'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                self.clip_labels = np.array(json.load(f))
        else:
            # Fallback: use zeros if labels not available
            self.clip_labels = np.zeros(
                (len(self.clip_features), len(CONDITIONS)), dtype=np.int64
            )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        # Load and transform image
        image_path = ann['image_path']
        image = Image.open(
            os.path.join(self.image_root, image_path[0])
        ).convert('RGB')
        image = self.transform(image)

        # Disease labels for this sample
        cls_labels = ann['labels']  # list of 14 ints {0,1,2,3}

        # Build diagnosis prompt prefix (same as PromptMRG)
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt) + ' '
        caption = prompt + my_pre_caption(ann['report'], self.max_words)

        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        # Retrieve top-k text features from bank
        k = self.args.clip_k
        clip_indices = ann['clip_indices'][:k]
        retrieved_text_feats = self.clip_features[clip_indices]
        retrieved_text_feats = torch.from_numpy(retrieved_text_feats).float()

        # Disease labels of retrieved reports (for DMCL and DCR)
        retrieved_text_labels = self.clip_labels[clip_indices[0]]  # top-1 labels
        retrieved_text_labels = torch.from_numpy(retrieved_text_labels).long()

        return image, caption, cls_labels, retrieved_text_feats, retrieved_text_labels


class DCAFEvalDataset(Dataset):
    """
    Evaluation dataset for DCAF (val/test).

    Args:
        transform: image transform pipeline
        image_root: root directory for images
        ann_root: path to annotation JSON file
        tokenizer: BertTokenizer
        max_words: maximum report length in words
        split: 'val' or 'test' (for MIMIC-CXR) or None (for IU X-Ray)
        dataset: 'mimic_cxr' or 'iu_xray'
        args: argument namespace
    """

    def __init__(
        self,
        transform,
        image_root: str,
        ann_root: str,
        tokenizer,
        max_words: int = 100,
        split: str = 'val',
        dataset: str = 'mimic_cxr',
        args=None,
    ):
        self.annotation = json.load(open(ann_root, 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else:
            # IU X-Ray: use entire dataset
            self.ann = self.annotation
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.dataset = dataset
        self.args = args

        # Load pre-computed CLIP text features
        feat_path = getattr(args, 'clip_feat_path', None) or \
            './data/mimic_cxr/clip_text_features.json'
        with open(feat_path, 'r') as f:
            self.clip_features = np.array(json.load(f))

        # Load disease labels for retrieved reports
        label_path = getattr(args, 'clip_label_path', None) or \
            './data/mimic_cxr/clip_text_labels.json'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                self.clip_labels = np.array(json.load(f))
        else:
            self.clip_labels = np.zeros(
                (len(self.clip_features), len(CONDITIONS)), dtype=np.int64
            )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        # Load and transform image
        image_path = ann['image_path']
        image = Image.open(
            os.path.join(self.image_root, image_path[0])
        ).convert('RGB')
        image = self.transform(image)

        # Ground-truth report (no prompt prefix for evaluation)
        caption = my_pre_caption(ann['report'], self.max_words)

        # Disease labels
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        # Retrieved text features
        k = self.args.clip_k
        clip_indices = ann['clip_indices'][:k]
        retrieved_text_feats = self.clip_features[clip_indices]
        retrieved_text_feats = torch.from_numpy(retrieved_text_feats).float()

        # Disease labels of top-1 retrieved report
        retrieved_text_labels = self.clip_labels[clip_indices[0]]
        retrieved_text_labels = torch.from_numpy(retrieved_text_labels).long()

        return image, caption, cls_labels, retrieved_text_feats, retrieved_text_labels
