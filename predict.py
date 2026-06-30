import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

from model import build_maskrcnn

# =================== CONFIGURATION ===================
CHECKPOINT_PATH = "models/best_ovl_5.pth"
SCORE_THRESHOLD = 0.5
CLASS_NAMES = {1: "Entrapped Particle", 2: "Free Particle"}
CLASS_COLORS = {1: (255, 0, 0), 2: (0, 255, 0)}

MASK_ALPHA = 0.4
FONT_SIZE = 40   # 🔥 INCREASED TEXT SIZE

# =================== MODEL ===================
def load_model(checkpoint_path, device):
    model = build_maskrcnn(simclr_path=None, num_classes=NUM_CLASSES)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print("✅ Loaded with weights_only=True")
    except:
        print("⚠️ weights_only failed, trying normal load...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    print("✅ Model ready")

    return model

# =================== PREDICTION ANNOTATION ===================
def annotate_image(image_np, output, score_thresh):
    from PIL import ImageDraw, ImageFont

    overlay = image_np.copy()
    h, w = image_np.shape[:2]

    boxes = output['boxes'].detach().cpu().numpy()
    scores = output['scores'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    masks = output['masks'].detach().cpu().numpy()

    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < score_thresh:
            continue

        base_color = CLASS_COLORS.get(int(label), (255, 255, 255))
        mask_color = tuple(int(c * 0.5) for c in base_color)

        m = mask[0]
        m = (m - m.min()) / (m.max() + 1e-6)
        m = cv2.resize(m, (w, h))

        for c in range(3):
            overlay[:, :, c] = (
                overlay[:, :, c] * (1 - 0.4 * m) +
                mask_color[c] * (0.4 * m)
            ).astype(np.uint8)

        m_bin = (m > 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, base_color, 2)

        x1, y1, x2, y2 = box
        sx, sy = w / IMG_SIZE, h / IMG_SIZE
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), base_color, 2)

    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        name = CLASS_NAMES.get(int(label), f"cls_{label}")
        box_color = CLASS_COLORS.get(int(label), (255, 255, 255))

        x1, y1, _, _ = box
        sx, sy = w / IMG_SIZE, h / IMG_SIZE
        x1, y1 = int(x1 * sx), int(y1 * sy)

        text = f"{name} {score:.2f}"

        # 🔥 Bigger background box
        draw.rectangle([x1, y1 - 50, x1 + 300, y1], fill=box_color)
        draw.text((x1 + 10, y1 - 45), text, fill="white", font=font)

    return np.array(result)

# =================== GT ANNOTATION ===================
def annotate_gt_image(image_np, label_path):
    from PIL import ImageDraw, ImageFont

    overlay = image_np.copy()
    h, w = image_np.shape[:2]

    if not os.path.exists(label_path):
        return overlay

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            cls_id = int(parts[0]) + 1
            coords = list(map(float, parts[1:]))

            polygon_x = [coords[i] * w for i in range(0, len(coords), 2)]
            polygon_y = [coords[i] * h for i in range(1, len(coords), 2)]

            annotations.append((cls_id, polygon_x, polygon_y))

    for cls_id, polygon_x, polygon_y in annotations:
        base_color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        mask_color = tuple(int(c * 0.5) for c in base_color)

        poly_pts = np.array(list(zip(polygon_x, polygon_y)), dtype=np.int32)
        m = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(m, [poly_pts], 1.0)

        for c in range(3):
            overlay[:, :, c] = (
                overlay[:, :, c] * (1 - 0.4 * m) +
                mask_color[c] * (0.4 * m)
            ).astype(np.uint8)

        m_bin = m.astype(np.uint8)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, base_color, 2)

        x_min, y_min = int(min(polygon_x)), int(min(polygon_y))
        x_max, y_max = int(max(polygon_x)), int(max(polygon_y))
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), base_color, 2)

    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    for cls_id, polygon_x, polygon_y in annotations:
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        box_color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        x1, y1 = int(min(polygon_x)), int(min(polygon_y))

        draw.rectangle([x1, y1 - 50, x1 + 300, y1], fill=box_color)
        draw.text((x1 + 10, y1 - 45), name, fill="white", font=font)

    return np.array(result)

# =================== PREDICT ===================
def predict(model, image_path, checkpoint_path=CHECKPOINT_PATH, label_path=None, score_thresh=SCORE_THRESHOLD):

    img_pil = Image.open(image_path).convert('RGB')
    orig_np = np.array(img_pil)

    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    img_tensor = transform(img_pil).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    annotated = annotate_image(orig_np, output, score_thresh)

    if label_path is None:
        base, _ = os.path.splitext(image_path)
        label_path = base + '.txt'

    gt_annotated = annotate_gt_image(orig_np, label_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(gt_annotated)
    ax1.set_title("Ground Truth", fontsize=16, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(annotated)
    ax2.set_title("Prediction", fontsize=16, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# =================== RUN ===================
NUM_CLASSES, IMG_SIZE = 3, 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(CHECKPOINT_PATH, device)

IMAGE_PATH = "dataset/images/A60-2-0-3-0-1.jpg"
LABEL_PATH = f"dataset/labels_txt/{IMAGE_PATH.split('/')[-1].split('.')[0]}.txt"

print(IMAGE_PATH)
predict(model, IMAGE_PATH, CHECKPOINT_PATH, LABEL_PATH, SCORE_THRESHOLD)