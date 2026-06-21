#!/bin/bash
# Training script for DCAF on MIMIC-CXR dataset
# Paper: "Disease-Consistent Visual-Text Alignment and Fusion for Reliable Radiology Report Generation"

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_NAME="mimic_cxr"
IMAGE_DIR="data/mimic_cxr/images/"
ANN_PATH="data/mimic_cxr/mimic_annotation_dcaf.json"
CLIP_FEAT_PATH="data/mimic_cxr/clip_text_features.json"
CLIP_LABEL_PATH="data/mimic_cxr/clip_text_labels.json"
SAVE_DIR="results/dcaf_mimic_cxr"

# Model hyperparameters (from paper)
EMBED_DIM=256
NUM_HEADS=8
SINKHORN_EPS=0.05
TEMPERATURE=0.07
CLIP_K=18

# Training hyperparameters (from paper)
BATCH_SIZE=32
EPOCHS=10
INIT_LR=5e-4
MIN_LR=5e-5
WARMUP_LR=5e-6
WEIGHT_DECAY=0.01
WARMUP_STEPS=2000

# Loss weights
CLS_WEIGHT=4.0
LAMBDA_MATCH=1.0
LAMBDA_CLS=1.0
LAMBDA_GEN=1.0

# Generation settings
BEAM_SIZE=3
GEN_MAX_LEN=150
GEN_MIN_LEN=100

# ── Single GPU Training ───────────────────────────────────────────────────────
python main_train.py \
    --dataset_name ${DATASET_NAME} \
    --image_dir ${IMAGE_DIR} \
    --ann_path ${ANN_PATH} \
    --clip_feat_path ${CLIP_FEAT_PATH} \
    --clip_label_path ${CLIP_LABEL_PATH} \
    --save_dir ${SAVE_DIR} \
    --embed_dim ${EMBED_DIM} \
    --num_heads ${NUM_HEADS} \
    --sinkhorn_eps ${SINKHORN_EPS} \
    --temperature ${TEMPERATURE} \
    --clip_k ${CLIP_K} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --init_lr ${INIT_LR} \
    --min_lr ${MIN_LR} \
    --warmup_lr ${WARMUP_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --cls_weight ${CLS_WEIGHT} \
    --lambda_match ${LAMBDA_MATCH} \
    --lambda_cls ${LAMBDA_CLS} \
    --lambda_gen ${LAMBDA_GEN} \
    --beam_size ${BEAM_SIZE} \
    --gen_max_len ${GEN_MAX_LEN} \
    --gen_min_len ${GEN_MIN_LEN} \
    --distributed False \
    --monitor_metric ce_f1

# ── Multi-GPU Training (uncomment to use) ────────────────────────────────────
# NPROC=4  # number of GPUs
# torchrun --nproc_per_node=${NPROC} main_train.py \
#     --dataset_name ${DATASET_NAME} \
#     --image_dir ${IMAGE_DIR} \
#     --ann_path ${ANN_PATH} \
#     --clip_feat_path ${CLIP_FEAT_PATH} \
#     --clip_label_path ${CLIP_LABEL_PATH} \
#     --save_dir ${SAVE_DIR} \
#     --embed_dim ${EMBED_DIM} \
#     --num_heads ${NUM_HEADS} \
#     --sinkhorn_eps ${SINKHORN_EPS} \
#     --temperature ${TEMPERATURE} \
#     --clip_k ${CLIP_K} \
#     --batch_size ${BATCH_SIZE} \
#     --epochs ${EPOCHS} \
#     --init_lr ${INIT_LR} \
#     --min_lr ${MIN_LR} \
#     --warmup_lr ${WARMUP_LR} \
#     --weight_decay ${WEIGHT_DECAY} \
#     --warmup_steps ${WARMUP_STEPS} \
#     --cls_weight ${CLS_WEIGHT} \
#     --lambda_match ${LAMBDA_MATCH} \
#     --lambda_cls ${LAMBDA_CLS} \
#     --lambda_gen ${LAMBDA_GEN} \
#     --beam_size ${BEAM_SIZE} \
#     --gen_max_len ${GEN_MAX_LEN} \
#     --gen_min_len ${GEN_MIN_LEN} \
#     --distributed True \
#     --monitor_metric ce_f1
