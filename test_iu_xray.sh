#!/bin/bash
# Test script for DCAF on IU X-Ray dataset
# Uses MIMIC-CXR pretrained model (following PromptMRG evaluation protocol)

DATASET_NAME="iu_xray"
IMAGE_DIR="data/iu_xray/images/"
ANN_PATH="data/iu_xray/annotation.json"
# Use MIMIC-CXR CLIP features for retrieval (cross-dataset evaluation)
CLIP_FEAT_PATH="data/mimic_cxr/clip_text_features.json"
CLIP_LABEL_PATH="data/mimic_cxr/clip_text_labels.json"
CHECKPOINT="results/dcaf_mimic_cxr/model_best.pth"  # MIMIC-CXR pretrained
SAVE_DIR="results/dcaf_iu_xray"

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
