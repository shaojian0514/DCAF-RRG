#!/bin/bash
# Test script for DCAF on MIMIC-CXR dataset

DATASET_NAME="mimic_cxr"
IMAGE_DIR="data/mimic_cxr/images/"
ANN_PATH="data/mimic_cxr/mimic_annotation_dcaf.json"
CLIP_FEAT_PATH="data/mimic_cxr/clip_text_features.json"
CLIP_LABEL_PATH="data/mimic_cxr/clip_text_labels.json"
CHECKPOINT="results/dcaf_mimic_cxr/model_best.pth"
SAVE_DIR="results/dcaf_mimic_cxr"

python main_test.py \
    --dataset_name ${DATASET_NAME} \
    --image_dir ${IMAGE_DIR} \
    --ann_path ${ANN_PATH} \
    --clip_feat_path ${CLIP_FEAT_PATH} \
    --clip_label_path ${CLIP_LABEL_PATH} \
    --load_pretrained ${CHECKPOINT} \
    --save_dir ${SAVE_DIR} \
    --batch_size 16 \
    --clip_k 18 \
    --beam_size 3 \
    --gen_max_len 150 \
    --gen_min_len 100 \
    --save_reports \
    --distributed False
