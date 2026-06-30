import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import IMG_SIZE, MULTI_SCALE_SIZES


def parse_yolo_annotation(label_path, img_w, img_h):
    annotations = []
    if not os.path.exists(label_path): return annotations
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7: continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = []
            for i in range(0, len(coords), 2):
                polygon.extend([coords[i]*img_w, coords[i+1]*img_h])
            xs, ys = polygon[0::2], polygon[1::2]
            x_min, y_min = min(xs), min(ys)
            w, h = max(xs)-x_min, max(ys)-y_min
            annotations.append({
                'category_id': cls_id+1, 'polygon': polygon,
                'bbox': [x_min,y_min,w,h], 'area': w*h
            })
    return annotations


def build_coco_json(image_files, images_dir, labels_dir):
    coco = {
        'info':{}, 'licenses':[], 'images':[], 'annotations':[],
        'categories':[{'id':1,'name':'entrap particle'},{'id':2,'name':'free particle'}]
    }
    ann_id = 1
    for img_id, fname in enumerate(image_files, start=1):
        img = Image.open(os.path.join(images_dir, fname))
        w, h = img.size
        coco['images'].append({'id':img_id,'file_name':fname,'width':w,'height':h})
        name = os.path.splitext(fname)[0]
        for ann in parse_yolo_annotation(os.path.join(labels_dir, name+'.txt'), w, h):
            coco['annotations'].append({
                'id':ann_id, 'image_id':img_id, 'category_id':ann['category_id'],
                'bbox':ann['bbox'], 'area':ann['area'],
                'segmentation':[ann['polygon']], 'iscrowd':0
            })
            ann_id += 1
    return coco


class ParticleDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root, self.transforms = root, transforms
        with open(ann_file) as f:
            self.coco = json.load(f)
        self.images = {img['id']:img for img in self.coco['images']}
        self.image_ids = list(self.images.keys())
        self.anns = {}
        for a in self.coco['annotations']:
            self.anns.setdefault(a['image_id'],[]).append(a)

    def __len__(self): return len(self.image_ids)

    def _poly2mask(self, poly, h, w):
        mask = Image.new('L',(w,h),0)
        pts = [(poly[i],poly[i+1]) for i in range(0,len(poly),2)]
        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
        return np.array(mask)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.images[img_id]
        img = Image.open(os.path.join(self.root, info['file_name'])).convert('RGB')
        anns = self.anns.get(img_id, [])
        boxes, labels, masks, areas = [], [], [], []
        for a in anns:
            x,y,w,h = a['bbox']
            if w<1 or h<1: continue
            boxes.append([x,y,x+w,y+h])
            labels.append(a['category_id'])
            areas.append(a['area'])
            masks.append(self._poly2mask(a['segmentation'][0], info['height'], info['width']))
        if boxes:
            target = dict(
                boxes=torch.as_tensor(boxes, dtype=torch.float32),
                labels=torch.as_tensor(labels, dtype=torch.int64),
                masks=torch.as_tensor(np.array(masks), dtype=torch.uint8),
                area=torch.as_tensor(areas, dtype=torch.float32),
                iscrowd=torch.zeros(len(boxes), dtype=torch.int64),
                image_id=torch.tensor([img_id])
            )
        else:
            target = dict(
                boxes=torch.zeros((0,4), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, info['height'], info['width']), dtype=torch.uint8),
                area=torch.zeros((0,), dtype=torch.float32),
                iscrowd=torch.zeros((0,), dtype=torch.int64),
                image_id=torch.tensor([img_id])
            )
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = T.ToTensor()(img)
        return img, target


# FIX 3: Copy-Paste augmentation for instance segmentation
def copy_paste_augment(dataset, img, target, paste_prob=0.5):
    """Randomly paste instances from another image onto the current one."""
    if random.random() > paste_prob or len(dataset) < 2:
        return img, target
    # Pick a random donor image
    donor_idx = random.randint(0, len(dataset) - 1)
    donor_img, donor_target = dataset[donor_idx]

    if len(donor_target['boxes']) == 0:
        return img, target

    # Convert tensors back if needed
    if isinstance(img, torch.Tensor):
        # img is C,H,W tensor
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    else:
        img_np = np.array(img)

    if isinstance(donor_img, torch.Tensor):
        donor_np = (donor_img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    else:
        donor_np = np.array(donor_img)

    h, w = img_np.shape[:2]
    dh, dw = donor_np.shape[:2]

    # Pick 1-2 random instances to paste
    n_paste = min(random.randint(1, 2), len(donor_target['boxes']))
    indices = random.sample(range(len(donor_target['boxes'])), n_paste)

    new_boxes = list(target['boxes'].numpy()) if len(target['boxes']) > 0 else []
    new_labels = list(target['labels'].numpy()) if len(target['labels']) > 0 else []
    new_masks = list(target['masks'].numpy()) if len(target['masks']) > 0 else []
    new_areas = list(target['area'].numpy()) if len(target['area']) > 0 else []

    for idx in indices:
        mask = donor_target['masks'][idx].numpy()
        box = donor_target['boxes'][idx].numpy()
        label = donor_target['labels'][idx].item()

        # Scale donor mask/box to current image size if needed
        if dh != h or dw != w:
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
            box[0] *= w / dw; box[2] *= w / dw
            box[1] *= h / dh; box[3] *= h / dh

        # Random offset
        max_dx = max(1, w // 4); max_dy = max(1, h // 4)
        dx, dy = random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)

        # Shift mask
        shifted_mask = np.zeros_like(mask)
        src_y1 = max(0, -dy); src_y2 = min(h, h - dy)
        src_x1 = max(0, -dx); src_x2 = min(w, w - dx)
        dst_y1 = max(0, dy); dst_y2 = min(h, h + dy)
        dst_x1 = max(0, dx); dst_x2 = min(w, w + dx)
        if dst_y2 > dst_y1 and dst_x2 > dst_x1 and src_y2 > src_y1 and src_x2 > src_x1:
            actual_h = min(dst_y2-dst_y1, src_y2-src_y1)
            actual_w = min(dst_x2-dst_x1, src_x2-src_x1)
            shifted_mask[dst_y1:dst_y1+actual_h, dst_x1:dst_x1+actual_w] = mask[src_y1:src_y1+actual_h, src_x1:src_x1+actual_w]

        if shifted_mask.sum() < 10:
            continue

        # Paste onto image
        paste_region = shifted_mask > 0
        # Scale donor image too
        if dh != h or dw != w:
            donor_resized = np.array(Image.fromarray(donor_np).resize((w, h), Image.BILINEAR))
        else:
            donor_resized = donor_np

        shifted_donor = np.zeros_like(img_np)
        if dst_y2 > dst_y1 and dst_x2 > dst_x1 and src_y2 > src_y1 and src_x2 > src_x1:
            shifted_donor[dst_y1:dst_y1+actual_h, dst_x1:dst_x1+actual_w] = donor_resized[src_y1:src_y1+actual_h, src_x1:src_x1+actual_w]

        img_np[paste_region] = shifted_donor[paste_region]

        # Update box from shifted mask
        ys, xs = np.where(shifted_mask > 0)
        new_box = [xs.min(), ys.min(), xs.max(), ys.max()]
        new_boxes.append(new_box)
        new_labels.append(label)
        new_masks.append(shifted_mask)
        new_areas.append((new_box[2]-new_box[0]) * (new_box[3]-new_box[1]))

    # Rebuild target
    if new_boxes:
        target = dict(
            boxes=torch.as_tensor(np.array(new_boxes, dtype=np.float32), dtype=torch.float32),
            labels=torch.as_tensor(np.array(new_labels, dtype=np.int64), dtype=torch.int64),
            masks=torch.as_tensor(np.array(new_masks), dtype=torch.uint8),
            area=torch.as_tensor(np.array(new_areas, dtype=np.float32), dtype=torch.float32),
            iscrowd=torch.zeros(len(new_boxes), dtype=torch.int64),
            image_id=target['image_id']
        )

    img = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
    return img, target


class ComposeT:
    def __init__(self, ts): self.ts = ts
    def __call__(self, image, target):
        ow, oh = image.size
        for t in self.ts:
            if isinstance(t, T.Resize):
                nh, nw = t.size
                image = t(image)
                if len(target['boxes'])>0:
                    sx,sy = nw/ow, nh/oh
                    target['boxes'][:,[0,2]] *= sx
                    target['boxes'][:,[1,3]] *= sy
                if len(target['masks'])>0:
                    ms = [np.array(Image.fromarray(m.numpy()).resize((nw,nh),Image.NEAREST)) for m in target['masks']]
                    target['masks'] = torch.as_tensor(np.array(ms), dtype=torch.uint8)
                ow, oh = nw, nh
            elif isinstance(t, T.RandomHorizontalFlip):
                if torch.rand(1)<0.5:
                    image = T.functional.hflip(image)
                    if len(target['boxes'])>0:
                        w = image.size[0] if not isinstance(image, torch.Tensor) else image.shape[-1]
                        target['boxes'][:,[0,2]] = w - target['boxes'][:,[2,0]]
                    if len(target['masks'])>0:
                        target['masks'] = target['masks'].flip(-1)
            elif isinstance(t, T.RandomVerticalFlip):
                if torch.rand(1)<0.5:
                    image = T.functional.vflip(image)
                    if len(target['boxes'])>0:
                        h = image.size[1] if not isinstance(image, torch.Tensor) else image.shape[-2]
                        target['boxes'][:,[1,3]] = h - target['boxes'][:,[3,1]]
                    if len(target['masks'])>0:
                        target['masks'] = target['masks'].flip(-2)
            elif isinstance(t, T.RandomAffine):
                # Apply affine to image, then manually adjust targets
                # For simplicity, apply rotation/scale to image only (masks handled via resize later)
                image = t(image)
            elif isinstance(t, T.ColorJitter):
                image = t(image)
            elif isinstance(t, T.GaussianBlur):
                image = t(image)
            elif isinstance(t, T.ToTensor):
                image = t(image)
        return image, target


def get_transform(train=True):
    if train:
        # FIX 3: Multi-scale training
        sz = random.choice(MULTI_SCALE_SIZES)
        ts = [T.Resize((sz, sz))]
        ts += [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
        ts.append(T.ToTensor())
    else:
        ts = [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()]
    return ComposeT(ts)


def collate_fn(batch): return tuple(zip(*batch))


print('Dataset classes ready (with Copy-Paste, Multi-Scale, stronger augmentation).')
