from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms

from effdet.anchors import Anchors


# post process -> cls_out, box_out, all_idx, all_cls


class EffdetPostProcess(nn.Module):
    def __init__(self, num_levels, num_classes, max_detection_points: int = 5000):
        super().__init__()
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.max_detection_points = max_detection_points

    def forward(self, cls_outputs: List[torch.Tensor], box_outputs: List[torch.Tensor]):
        batch_size = cls_outputs[0].shape[0]
        cls_outputs_all = torch.cat([
            cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, self.num_classes])
            for level in range(self.num_levels)], 1)

        box_outputs_all = torch.cat([
            box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
            for level in range(self.num_levels)], 1)

        _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=self.max_detection_points)
        indices_all = cls_topk_indices_all // self.num_classes
        classes_all = cls_topk_indices_all % self.num_classes

        box_outputs_all_after_topk = torch.gather(
            box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

        cls_outputs_all_after_topk = torch.gather(
            cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, self.num_classes))
        cls_outputs_all_after_topk = torch.gather(
            cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

        return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all
