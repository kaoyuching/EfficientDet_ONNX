from typing import Dict, Optional
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms

from effdet.anchors import Anchors


class EffdetNMS(nn.Module):
    def __init__(
        self,
        model_config,
        max_det_per_image: int = 100,
        soft_nms: bool = False
    ):
        super().__init__()
        self.max_det_per_image = max_det_per_image
        self.soft_nms = soft_nms
        self.anchors = Anchors.from_config(model_config)

    def decode_box_outputs(self, rel_codes, anchors, output_xyxy: bool = False):
        ycenter_a = (anchors[:, 0] + anchors[:, 2]) / 2
        xcenter_a = (anchors[:, 1] + anchors[:, 3]) / 2
        ha = anchors[:, 2] - anchors[:, 0]
        wa = anchors[:, 3] - anchors[:, 1]

        ty, tx, th, tw = rel_codes.unbind(dim=1)

        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        if output_xyxy:
            out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        else:
            out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
        return out

    def clip_boxes_xyxy(self, boxes: torch.Tensor, size: torch.Tensor):
        boxes = boxes.clamp(min=0)
        size = torch.cat([size, size], dim=0)
        boxes = boxes.min(size)
        return boxes

    def generate_detections(
        self,
        cls_outputs,
        box_outputs,
        anchor_boxes,
        indices,
        classes,
        img_scale: Optional[torch.Tensor] = None,
        img_size: Optional[torch.Tensor] = None,
        soft_nms: bool = False,
    ):
        anchor_boxes = anchor_boxes[indices, :]

        # convert box to xyxy
        boxes = self.decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
        if img_scale is not None and img_size is not None:
            boxes = self.clip_boxes_xyxy(boxes, img_size / img_scale)

        scores = cls_outputs.sigmoid().squeeze(1).float()
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)

        top_detection_idx = top_detection_idx[:self.max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1

        if img_scale is not None:
            boxes = boxes * img_scale

        num_det = len(top_detection_idx)
        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        if num_det < self.max_det_per_image:
            detections = torch.cat([
                detections,
                torch.zeros((self.max_det_per_image - num_det, 6))
            ], dim=0)
        return detections

    def forward(self, cls_outputs, box_outputs, indices, classes, img_info: Optional[Dict[str, torch.Tensor]] = None):
        r"""
        Return:
            shape: (batch, max_det_per_image, 6)
                each row representing [x_min, y_min, x_max, y_max, score, class]
        """
        if img_info is None:
            img_scale, img_size = None, None
        else:
            img_scale, img_size = img_info['img_scale'], img_info['img_size']

        batch_size = cls_outputs.shape[0]
        batch_detections = []
        for i in range(batch_size):
            img_scale_i = None if img_scale is None else img_scale[i]
            img_size_i = None if img_size is None else img_size[i]
            detection = self.generate_detections(
                cls_outputs[i],
                box_outputs[i],
                self.anchors.boxes,
                indices[i],
                classes[i],
                img_scale_i,
                img_size_i,
            )
            batch_detections.append(detection)
        return torch.stack(batch_detections, dim=0)
