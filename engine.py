import sys
import io

import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import IMG_SIZE
from dataset import copy_paste_augment


def train_one_epoch(model, loader, optimizer, device, epoch, dataset, ema=None):
    model.train()
    total_loss, n = 0.0, 0
    for i, (imgs, targets) in enumerate(loader):
        imgs = list(imgs); targets = list(targets)

        # FIX 3: Apply copy-paste augmentation on-the-fly
        augmented = []
        for img, tgt in zip(imgs, targets):
            img_cp, tgt_cp = copy_paste_augment(dataset, img, tgt, paste_prob = 0.0 if epoch < 5 else 0.05 )
            augmented.append((img_cp, tgt_cp))
        imgs_clean, targets_clean = [], []
        for a in augmented:
            img_a, tgt_a = a[0], a[1]
            # V5 FIX: Filter degenerate boxes (zero width/height) to prevent AssertionError
            if len(tgt_a['boxes']) > 0:
                boxes = tgt_a['boxes']
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                valid = (w > 1) & (h > 1)
                if valid.sum() > 0:
                    tgt_a = {k: v[valid] if k not in ('image_id',) and v.shape[0] == len(boxes) else v for k, v in tgt_a.items()}
                else:
                    tgt_a = dict(
                        boxes=torch.zeros((0,4), dtype=torch.float32),
                        labels=torch.zeros((0,), dtype=torch.int64),
                        masks=torch.zeros((0, img_a.shape[1], img_a.shape[2]), dtype=torch.uint8),
                        area=torch.zeros((0,), dtype=torch.float32),
                        iscrowd=torch.zeros((0,), dtype=torch.int64),
                        image_id=tgt_a['image_id']
                    )
            imgs_clean.append(img_a.to(device))
            targets_clean.append({k:v.to(device) for k,v in tgt_a.items()})
        imgs, targets = imgs_clean, targets_clean

        # Skip batch if all targets are empty
        if all(len(t['boxes']) == 0 for t in targets):
            continue

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            optimizer.zero_grad(); continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item(); n += 1
        """if (i+1)%10==0 or i==len(loader)-1:
            print(f'  [{i+1}/{len(loader)}] loss={loss.item():.4f} '
                  f'(cls={loss_dict.get("loss_classifier",0):.3f} '
                  f'box={loss_dict.get("loss_box_reg",0):.3f} '
                  f'mask={loss_dict.get("loss_mask",0):.3f})', flush=True)"""
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_map50(model, loader, device, ann_file, iou_type='bbox'):
    """Compute mAP@50 with V4 fixes: no score filtering, detection count logging."""
    model.eval()
    coco_gt = COCO(ann_file)
    if 'info' not in coco_gt.dataset: coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset: coco_gt.dataset['licenses'] = []
    img_sizes = {img['id']:(img['width'],img['height']) for img in coco_gt.dataset['images']}
    results = []
    total_dets = 0
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for tgt, out in zip(targets, outputs):
            img_id = tgt['image_id'].item()
            ow, oh = img_sizes[img_id]
            sx, sy = ow/IMG_SIZE, oh/IMG_SIZE
            n_dets = len(out['boxes'])
            total_dets += n_dets
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
    if not results: return 0.0
    # Suppress verbose COCO eval output to prevent Jupyter output truncation
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        ap50 = ev.stats[1]  # AP @ IoU=0.50
    except Exception:
        ap50 = 0.0
    finally:
        sys.stdout = old_stdout
    return ap50


@torch.no_grad()
def evaluate_detailed(model, loader, device, ann_file, iou_type='bbox'):
    """Compute overall mAP@50 and per-class AP@50."""
    model.eval()
    coco_gt = COCO(ann_file)
    if 'info' not in coco_gt.dataset: coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset: coco_gt.dataset['licenses'] = []
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
    if not results:
        return 0.0, 0.0, 0.0
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(results)
        def get_ap50(cat_ids=None):
            ev = COCOeval(coco_gt, coco_dt, iou_type)
            ev.params.iouThrs = [0.5]
            if cat_ids: ev.params.catIds = cat_ids
            ev.evaluate(); ev.accumulate(); ev.summarize()
            return ev.stats[0] if len(ev.stats) > 0 else 0.0
        overall_ap50 = get_ap50()
        cat1_ap50 = get_ap50(cat_ids=[1])  # entrapped particle
        cat2_ap50 = get_ap50(cat_ids=[2])  # free particle
    except Exception:
        overall_ap50, cat1_ap50, cat2_ap50 = 0.0, 0.0, 0.0
    finally:
        sys.stdout = old_stdout
    return overall_ap50, cat1_ap50, cat2_ap50


print('Training helpers ready.')
