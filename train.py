import os
import sys
import json
import time
import random
import math
import copy
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timm.utils import ModelEma

from config import *
from model import build_maskrcnn, freeze_backbone_stages
from dataset import build_coco_json, ParticleDataset, get_transform, collate_fn
from engine import train_one_epoch, evaluate_map50, evaluate_detailed


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # =================== BUILD MODEL ===================
    model = build_maskrcnn(SIMCLR_CHECKPOINT, NUM_CLASSES)
    freeze_backbone_stages(model.backbone)
    model.to(device)
    ema = ModelEma(model, decay=EMA_DECAY)

    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {total:,} total | {train_p:,} trainable | {total-train_p:,} frozen')

    # =================== DATA PREPARATION ===================
    all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    paired = [f for f in all_images if os.path.exists(os.path.join(LABELS_DIR, os.path.splitext(f)[0]+'.txt'))]
    random.seed(RANDOM_SEED); random.shuffle(paired)
    n_total = len(paired)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    train_files = paired[:n_train]
    test_files = paired[n_train:n_train + n_val]
    val_files = paired[n_train + n_val:]

    train_coco = build_coco_json(train_files, IMAGES_DIR, LABELS_DIR)
    val_coco = build_coco_json(val_files, IMAGES_DIR, LABELS_DIR)
    test_coco = build_coco_json(test_files, IMAGES_DIR, LABELS_DIR)
    TRAIN_ANN = os.path.join(ANNOTATIONS_DIR, 'train.json')
    VAL_ANN = os.path.join(ANNOTATIONS_DIR, 'val.json')
    TEST_ANN = os.path.join(ANNOTATIONS_DIR, 'test.json')
    with open(TRAIN_ANN,'w') as f: json.dump(train_coco, f)
    with open(VAL_ANN,'w') as f: json.dump(val_coco, f)
    with open(TEST_ANN,'w') as f: json.dump(test_coco, f)
    print(f'Train: {len(train_files)} imgs, {len(train_coco["annotations"])} anns')
    print(f'Val: {len(val_files)} imgs, {len(val_coco["annotations"])} anns')
    print(f'Test: {len(test_files)} imgs, {len(test_coco["annotations"])} anns')

    # =================== DATA LOADERS ===================
    train_ds = ParticleDataset(IMAGES_DIR, TRAIN_ANN, get_transform(train=True))
    val_ds = ParticleDataset(IMAGES_DIR, VAL_ANN, get_transform(train=False))
    test_ds = ParticleDataset(IMAGES_DIR, TEST_ANN, get_transform(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # FIX 5: Differential learning rates
    backbone_params = [p for p in model.backbone.features.parameters() if p.requires_grad]
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f'Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs')
    print(f'Optimizer: AdamW backbone_lr={BACKBONE_LR} head_lr={HEAD_LR} wd={WEIGHT_DECAY}')
    print(f'Scheduler: cosine with {WARMUP_EPOCHS}-epoch warmup')
    print(f'Multi-scale training sizes: {MULTI_SCALE_SIZES}')
    print(f'Backbone params: {sum(p.numel() for p in backbone_params):,}')
    print(f'Head/neck params: {sum(p.numel() for p in head_params):,}')

    # =================== TRAINING LOOP ===================
    print('='*60)
    print('Starting Training (V6 — Always Evaluate Trained Model)')
    print('='*60)
    sys.stdout.flush()

    best_score = 0.0
    best_box_map50 = 0.0
    best_mask_map50 = 0.0
    best_box_epoch = 0
    best_mask_epoch = 0
    history = {'loss': [], 'box_map50': [], 'mask_map50': [], 'lr_backbone': [], 'lr_head': []}

    for epoch in range(1, EPOCHS + 1):
        # Multi-scale training — recreate transforms each epoch
        train_ds.transforms = get_transform(train=True)

        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        print(f'\nEpoch [{epoch}/{EPOCHS}]  lr_backbone={lr_bb:.6f}  lr_head={lr_hd:.6f}', flush=True)

        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, train_ds, ema=ema)
        t1 = time.time()
        print(f'  Train loss: {avg_loss:.4f}  ({t1-t0:.0f}s)', flush=True)

        # V5 FIX: Always evaluate with the TRAINED model (EMA was collapsing metrics)
        box_map50 = evaluate_map50(model, val_loader, device, VAL_ANN, 'bbox')
        mask_map50 = evaluate_map50(model, val_loader, device, VAL_ANN, 'segm')
        print(f'  Val Box  mAP@50: {box_map50:.4f}', flush=True)
        print(f'  Val Mask mAP@50: {mask_map50:.4f}', flush=True)

        history['loss'].append(avg_loss)
        history['box_map50'].append(box_map50)
        history['mask_map50'].append(mask_map50)
        history['lr_backbone'].append(lr_bb)
        history['lr_head'].append(lr_hd)

        scheduler.step()

        # Update EMA (keep it updated for potential future use, but don't eval with it)
        ema.update(model)

        score = (box_map50 + mask_map50) / 2
        both_above_80 = (box_map50 > 0.80) and (mask_map50 > 0.80)
        if score > best_score:
            best_score = score
            path = os.path.join(CHECKPOINT_DIR,'best_ovl.pth')
            torch.save({
                'epoch':epoch, 'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'box_map50':box_map50, 'mask_map50':mask_map50, 'loss':avg_loss
            }, path)
            print(f'  \u2605 Best model saved (avg={score:.4f})', flush=True)

        if box_map50 > best_box_map50:
            path = os.path.join(CHECKPOINT_DIR,'best_box_map50.pth')
            torch.save({
                'epoch':epoch, 'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'box_map50':box_map50, 'mask_map50':mask_map50, 'loss':avg_loss
            }, path)
            print(f'  \u2605 Best box_map50 model saved (avg={score:.4f})', flush=True)

        if mask_map50 > best_mask_map50:
            path = os.path.join(CHECKPOINT_DIR, 'best_mask_map50.pth')
            torch.save({
                'epoch':epoch, 'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'box_map50':box_map50, 'mask_map50':mask_map50, 'loss':avg_loss
            }, path)
            print(f'  \u2605 Best mask_map50 model saved (avg={score:.4f})', flush=True)
        # Track best individual metrics
        if box_map50 > best_box_map50:
            best_box_map50 = box_map50
            best_box_epoch = epoch
        if mask_map50 > best_mask_map50:
            best_mask_map50 = mask_map50
            best_mask_epoch = epoch

        sys.stdout.flush()
    path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    torch.save({
        'epoch':epoch, 'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'box_map50':box_map50, 'mask_map50':mask_map50, 'loss':avg_loss
    }, path)
    print(f'  \u2605 final model saved (avg={score:.4f})', flush=True)
    print(f'\nTraining complete. Best avg mAP@50: {best_score:.4f}', flush=True)
    print(f'Best Box  mAP@50: {best_box_map50:.4f} (epoch {best_box_epoch})', flush=True)
    print(f'Best Mask mAP@50: {best_mask_map50:.4f} (epoch {best_mask_epoch})', flush=True)

    # ---- Final Test Evaluation ----
    print('\n' + '='*60)
    print('Running final evaluation on TEST set...')
    print('='*60)
    test_box_map50, box_ent, box_fre = evaluate_detailed(model, test_loader, device, TEST_ANN, 'bbox')
    test_mask_map50, msk_ent, msk_fre = evaluate_detailed(model, test_loader, device, TEST_ANN, 'segm')
    print(f'  Test Box  mAP@50           : {test_box_map50:.4f}', flush=True)
    print(f'  Test Mask mAP@50           : {test_mask_map50:.4f}', flush=True)
    print(f'  AP50 Box  Entrapped Particle: {box_ent:.4f}', flush=True)
    print(f'  AP50 Box  Free Particle     : {box_fre:.4f}', flush=True)
    print(f'  AP50 Mask Entrapped Particle: {msk_ent:.4f}', flush=True)
    print(f'  AP50 Mask Free Particle     : {msk_fre:.4f}', flush=True)
    print(f'  Test Avg  mAP@50           : {(test_box_map50 + test_mask_map50) / 2:.4f}', flush=True)

    # =================== SAVE TRAINING STATES TO JSON ===================
    import json as _json

    training_states = {
        'config': {
            'num_classes': NUM_CLASSES,
            'train_split': TRAIN_SPLIT,
            'random_seed': RANDOM_SEED,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'backbone_lr': BACKBONE_LR,
            'head_lr': HEAD_LR,
            'weight_decay': WEIGHT_DECAY,
            'nms_iou_thresh': NMS_IOU_THRESH,
            'img_size': IMG_SIZE,
            'warmup_epochs': WARMUP_EPOCHS,
            'ema_decay': EMA_DECAY,
            'multi_scale_sizes': MULTI_SCALE_SIZES,
        },
        'history': {
            'loss': history['loss'],
            'box_map50': history['box_map50'],
            'mask_map50': history['mask_map50'],
            'lr_backbone': history['lr_backbone'],
            'lr_head': history['lr_head'],
        },
        'best_results': {
            'best_avg_map50': best_score,
            'best_box_map50': best_box_map50,
            'best_box_epoch': best_box_epoch,
            'best_mask_map50': best_mask_map50,
            'best_mask_epoch': best_mask_epoch,
        },
        'test_results': {
            'test_box_map50': test_box_map50,
            'test_mask_map50': test_mask_map50,
            'test_avg_map50': (test_box_map50 + test_mask_map50) / 2,
            'AP50_box_entrapped': box_ent,
            'AP50_box_free': box_fre,
            'AP50_mask_entrapped': msk_ent,
            'AP50_mask_free': msk_fre,
        },
        'dataset_info': {
            'train_images': len(train_files),
            'val_images': len(val_files),
            'test_images': len(test_files),
            'train_annotations': len(train_coco['annotations']),
            'val_annotations': len(val_coco['annotations']),
            'test_annotations': len(test_coco['annotations']),
        },
    }

    states_path = os.path.join(CHECKPOINT_DIR, 'training_states.json')
    with open(states_path, 'w') as _f:
        _json.dump(training_states, _f, indent=2)
    print(f'\nTraining states saved to: {states_path}', flush=True)

    # =================== PLOT TRAINING CURVES ===================
    epochs_range = range(1, len(history['loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs_range, history['loss'], 'b-o', markersize=3, linewidth=1.5)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # mAP@50
    axes[1].plot(epochs_range, history['box_map50'], 'r-o', markersize=3, linewidth=1.5, label='Box mAP@50')
    axes[1].plot(epochs_range, history['mask_map50'], 'g-o', markersize=3, linewidth=1.5, label='Mask mAP@50')
    axes[1].set_title('Validation mAP@50', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mAP@50')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs_range, history['lr_backbone'], 'purple', linewidth=1.5, label='Backbone LR')
    axes[2].plot(epochs_range, history['lr_head'], 'orange', linewidth=1.5, label='Head LR')
    axes[2].set_title('Learning Rate Schedule', fontsize=14)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'v6_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\nBest Box  mAP@50: {best_box_map50:.4f} (epoch {best_box_epoch})')
    print(f'Best Mask mAP@50: {best_mask_map50:.4f} (epoch {best_mask_epoch})')
    print(f'Best Avg  mAP@50: {best_score:.4f}')
    print(f'Completed epochs: {len(history["loss"])}')
