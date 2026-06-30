# =================================================================
# EXACT PAPER REPLICATION: Table B.1 (Train/Val/Test + 3 Runs)
# =================================================================
import json
import os
import sys
import io
import time
import copy
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import *
from model import build_maskrcnn, freeze_backbone_stages
from dataset import (
    build_coco_json, ParticleDataset, get_transform, collate_fn
)
from engine import train_one_epoch, evaluate_map50


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*80}\nSTARTING EXACT PAPER REPLICATION (STAGE 2)\n{'='*80}")

    # 1. EXACT PAPER DATA SPLIT (80% Train, 10% Val, 10% Test)
    all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    paired = [f for f in all_images if os.path.exists(os.path.join(LABELS_DIR, os.path.splitext(f)[0]+'.txt'))]
    random.seed(RANDOM_SEED)
    random.shuffle(paired)
    n_total = len(paired)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    base_train_files = paired[:n_train]
    base_val_files = paired[n_train:n_train+n_val]
    test_files = paired[n_train+n_val:]

    print(f"Data Base Split: {len(base_train_files)} Train | {len(base_val_files)} Val | {len(test_files)} Test")

    # Build and save static Test Dataset (Never shrinks)
    test_coco = build_coco_json(test_files, IMAGES_DIR, LABELS_DIR)
    TEST_ANN = os.path.join(ANNOTATIONS_DIR, 'test_static.json')
    with open(TEST_ANN, 'w') as f: json.dump(test_coco, f)
    test_ds = ParticleDataset(IMAGES_DIR, TEST_ANN, get_transform(train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. DETAILED EVALUATION HELPER (Calculates overall mAP50 and per-class AP50)
    @torch.no_grad()
    def evaluate_detailed(model, loader, device, ann_file, iou_type='bbox'):
        model.eval()
        coco_gt = COCO(ann_file)
        img_sizes = {img['id']:(img['width'],img['height']) for img in coco_gt.dataset['images']}
        results = []

        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for tgt, out in zip(targets, outputs):
                img_id = tgt['image_id'].item()
                ow, oh = img_sizes[img_id]
                sx, sy = ow/IMG_SIZE, oh/IMG_SIZE

                if iou_type == 'bbox':
                    for box, sc, lb in zip(out['boxes'].cpu().numpy(), out['scores'].cpu().numpy(), out['labels'].cpu().numpy()):
                        x1,y1,x2,y2 = box
                        results.append({
                            'image_id':int(img_id), 'category_id':int(lb),
                            'bbox':[float(x1*sx),float(y1*sy),float((x2-x1)*sx),float((y2-y1)*sy)],
                            'score':float(sc)
                        })
                else:
                    import pycocotools.mask as mask_util
                    for mask_pred, sc, lb in zip(out['masks'].cpu(), out['scores'].cpu().numpy(), out['labels'].cpu().numpy()):
                        m = (mask_pred[0] > 0.5).numpy().astype(np.uint8)
                        m_resized = np.array(Image.fromarray(m).resize((ow,oh), Image.NEAREST))
                        rle = mask_util.encode(np.asfortranarray(m_resized))
                        rle['counts'] = rle['counts'].decode('utf-8')
                        results.append({
                            'image_id':int(img_id), 'category_id':int(lb),
                            'segmentation':rle, 'score':float(sc)
                        })

        if not results: return 0.0, 0.0, 0.0

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            coco_dt = coco_gt.loadRes(results)

            # Function to run COCOeval for specific parameters
            def get_ap50(cat_ids=None):
                ev = COCOeval(coco_gt, coco_dt, iou_type)
                ev.params.iouThrs = [0.5] # Restrict to AP50
                if cat_ids: ev.params.catIds = cat_ids
                ev.evaluate(); ev.accumulate(); ev.summarize()
                return ev.stats[0] if len(ev.stats) > 0 else 0.0

            overall_ap50 = get_ap50()
            cat1_ap50 = get_ap50(cat_ids=[1]) # Entrapped
            cat2_ap50 = get_ap50(cat_ids=[2]) # Free

        except Exception:
            overall_ap50, cat1_ap50, cat2_ap50 = 0.0, 0.0, 0.0
        finally:
            sys.stdout = old_stdout

        return overall_ap50, cat1_ap50, cat2_ap50


    # 3. MAIN EVALUATION LOOP
    proportions = [0.1,]
    num_runs = 1
    final_table_data = {}

    for prop in proportions:
        prop_name = f"Train_{int(prop*100)}%"
        run_metrics = []

        # Subset Train & Val
        n_tr = int(len(base_train_files) * prop)
        n_vl = max(1, int(len(base_val_files) * prop)) # Ensure at least 1 val image

        sub_tr_files = base_train_files[:n_tr]
        sub_vl_files = base_val_files[:n_vl]

        TR_ANN = os.path.join(ANNOTATIONS_DIR, f"{prop_name}_train.json")
        VL_ANN = os.path.join(ANNOTATIONS_DIR, f"{prop_name}_val.json")

        with open(TR_ANN, 'w') as f: json.dump(build_coco_json(sub_tr_files, IMAGES_DIR, LABELS_DIR), f)
        with open(VL_ANN, 'w') as f: json.dump(build_coco_json(sub_vl_files, IMAGES_DIR, LABELS_DIR), f)

        for run in range(1, num_runs + 1):
            print(f"\n>>> {prop_name} | Run {run}/{num_runs} (Train: {len(sub_tr_files)}, Val: {len(sub_vl_files)})")

            # Datasets & Loaders
            tr_ds = ParticleDataset(IMAGES_DIR, TR_ANN, get_transform(train=True))
            vl_ds = ParticleDataset(IMAGES_DIR, VL_ANN, get_transform(train=False))
            tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

            # Init Model with SimCLR weights
            current_model = build_maskrcnn(SIMCLR_CHECKPOINT, NUM_CLASSES)
            freeze_backbone_stages(current_model.backbone)
            current_model.to(device)
            optimizer = torch.optim.AdamW(current_model.parameters(), lr=HEAD_LR)

            best_val_box = 0.0
            best_model_state = None

            # Track training time
            start_time = time.time()

            # Training Phase (100 Epochs)
            for epoch in range(1, 101):
                print(f"epoch: {epoch} ",end="\t")
                train_one_epoch(current_model, tr_loader, optimizer, device, epoch, tr_ds)

                # Validate to find the best model (using standard evaluate_map50 for speed)
                val_box = evaluate_map50(current_model, vl_loader, device, VL_ANN, 'bbox')
                if val_box > best_val_box:
                    best_val_box = val_box
                    best_model_state = copy.deepcopy(current_model.state_dict())

            # Calculate elapsed training time in minutes
            fine_tuning_time_min = int((time.time() - start_time) / 60)

            # Testing Phase (Load best weights, evaluate on STATIC TEST SET)
            current_model.load_state_dict(best_model_state)
            current_model.eval()

            # Detailed Evaluation
            box_ovr, box_ent, box_fre = evaluate_detailed(current_model, test_loader, device, TEST_ANN, 'bbox')
            msk_ovr, msk_ent, msk_fre = evaluate_detailed(current_model, test_loader, device, TEST_ANN, 'segm')

            run_metrics.append({
                'time_min': fine_tuning_time_min,
                'val_box': best_val_box,
                'test_box': box_ovr, 'test_mask': msk_ovr,
                'box_ent': box_ent, 'box_fre': box_fre,
                'msk_ent': msk_ent, 'msk_fre': msk_fre
            })

            print(f"    Run {run} TEST Results -> Box mAP: {box_ovr*100:.1f}%, Mask mAP: {msk_ovr*100:.1f}%, Time: {fine_tuning_time_min} mins")

        # =================================================================
        # BUILD NESTED JSON STRUCTURE
        # =================================================================
        final_table_data[prop_name] = {
            "runs": {},
            "average": {}
        }

        # 1. Populate individual runs
        for idx, m in enumerate(run_metrics, start=1):
            final_table_data[prop_name]["runs"][str(idx)] = {
                "fine_tuning_time_min": m['time_min'],
                "validation_mAP50_box": f"{m['val_box']*100:.1f}%",
                "test_mAP50_box": f"{m['test_box']*100:.1f}%",
                "test_mAP50_mask": f"{m['test_mask']*100:.1f}%",
                "AP50_box_entrapped": f"{m['box_ent']*100:.1f}%",
                "AP50_box_free": f"{m['box_fre']*100:.1f}%",
                "AP50_mask_entrapped": f"{m['msk_ent']*100:.1f}%",
                "AP50_mask_free": f"{m['msk_fre']*100:.1f}%"
            }

        # 2. Populate the computed average across those runs
        avg_metrics = {k: np.mean([m[k] for m in run_metrics]) for k in run_metrics[0].keys()}
        final_table_data[prop_name]["average"] = {
            "fine_tuning_time_min": int(avg_metrics['time_min']),
            "validation_mAP50_box": f"{avg_metrics['val_box']*100:.1f}%",
            "test_mAP50_box": f"{avg_metrics['test_box']*100:.1f}%",
            "test_mAP50_mask": f"{avg_metrics['test_mask']*100:.1f}%",
            "AP50_box_entrapped": f"{avg_metrics['box_ent']*100:.1f}%",
            "AP50_box_free": f"{avg_metrics['box_fre']*100:.1f}%",
            "AP50_mask_entrapped": f"{avg_metrics['msk_ent']*100:.1f}%",
            "AP50_mask_free": f"{avg_metrics['msk_fre']*100:.1f}%"
        }

        print(f"\n{'-'*60}\nRESULTS FOR {prop_name} STORED.\n{'-'*60}")

    # Save final Table B.1 JSON
    table_path = os.path.join(CHECKPOINT_DIR, 'table_b1_results_nested.json')
    with open(table_path, 'w') as f:
        json.dump(final_table_data, f, indent=4)

    print(f"\nAll operations complete! Exact Table B.1 replication saved to {table_path}")
