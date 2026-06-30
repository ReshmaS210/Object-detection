import os
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from config import NMS_IOU_THRESH, IMG_SIZE, MULTI_SCALE_SIZES
from backbone import SwinV2_AGPU_HFAM_FPN


def load_simclr_weights(backbone, path):
    """Load SimCLR pretrained weights into the Swin backbone."""
    print(f'Loading SimCLR weights from {path}')
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    # Try multiple key prefix patterns
    enc = {k:v for k,v in sd.items() if k.startswith('features.') or k.startswith('norm.')}
    if not enc:
        enc = {k[len('encoder.'):]:v for k,v in sd.items()
               if k.startswith('encoder.') and (k[len('encoder.'):].startswith('features.') or k[len('encoder.'):].startswith('norm.'))}
    if not enc:
        enc = sd
    cur = backbone.state_dict()
    loaded = 0
    for k, v in enc.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            loaded += 1
    backbone.load_state_dict(cur, strict=False)
    print(f'  Loaded {loaded} tensors')
    return backbone


def build_maskrcnn(simclr_path, num_classes=3):
    backbone = SwinV2_AGPU_HFAM_FPN(out_channels=256)

    # Load SimCLR weights (user requirement: keep SimCLR pretrained)
    if simclr_path and os.path.exists(simclr_path):
        backbone = load_simclr_weights(backbone, simclr_path)
    else:
        print(f'  WARNING: {simclr_path} not found — using random init')

    # FIX 4: Smaller anchors for particle detection — 4 FPN levels
    # anchor_gen = AnchorGenerator(
    #     sizes=((8, 16), (32, 64), (64, 128), (128, 256)),
    #     aspect_ratios=((0.5, 1.0, 2.0),) * 4
    # )
    anchor_gen = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    box_roi = MultiScaleRoIAlign(['0','1','2','3'], output_size=7, sampling_ratio=2)
    mask_roi = MultiScaleRoIAlign(['0','1','2','3'], output_size=28, sampling_ratio=2)

    return MaskRCNN(
        backbone, num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=box_roi, mask_roi_pool=mask_roi,
        box_nms_thresh=NMS_IOU_THRESH,
        box_score_thresh=0.0,  # V4 FIX: don't filter preds; let COCOeval handle it
        box_detections_per_img=100,
        min_size=IMG_SIZE,  # V4 FIX: match val image size
        max_size=max(MULTI_SCALE_SIZES) + 128,  # V4 FIX: allow headroom
        # Increase RPN proposals for better recall
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
    )


def freeze_backbone_stages(backbone):
    """FIX 2: Freeze stages 0-5 (patch embed + stages 1-3), only train stage 4 + neck."""
    for i in range(6):  # features[0]..features[5]
        for param in backbone.features[i].parameters():
            param.requires_grad = False
    print('Frozen: features[0..5] (Patch Embed + Stages 1-3)')
    print('Trainable: features[6..7] (Stage 4) + AGPU/HFAM neck + detection heads')


print('Model builder ready.')
