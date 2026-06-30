import os

# =================== CONFIGURATION ===================
SIMCLR_CHECKPOINT = "models/best_ovl_5.pth"
IMAGES_DIR        = "dataset/images"
LABELS_DIR        = "dataset/labels_txt"
ANNOTATIONS_DIR   = "annotations"
CHECKPOINT_DIR    = "checkpoints"

NUM_CLASSES    = 3
TRAIN_SPLIT    = 0.8
VAL_SPLIT      = 0.1
TEST_SPLIT     = 0.1
RANDOM_SEED    = 42
EPOCHS         = 100
BATCH_SIZE     = 4
BACKBONE_LR    = 1e-5        # FIX 5: Differential LR — backbone
HEAD_LR        = 5e-4        # FIX 5: Differential LR — heads/neck
WEIGHT_DECAY   = 0.005
NMS_IOU_THRESH = 0.5
IMG_SIZE       = 640
WARMUP_EPOCHS  = 10
EMA_DECAY      = 0.998       # V4 FIX: faster EMA warmup (was 0.9988)      # NEW: EMA smoothing

# Multi-scale training sizes (FIX 3)
MULTI_SCALE_SIZES = [576, 608, 640, 672, 704]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
print('Config loaded.')
