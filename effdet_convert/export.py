import os
from typing import Dict, Optional
import numpy as np
import torch

from effdet import EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from effdet import create_dataset


def get_model_config(model_name, num_classes: Optional[int] = None, img_size: Optional[int] = None):
    r"""
    Example: model_name="tf_efficientdet_d0", num_classes=1, img_size=512
    """
    config = get_efficientdet_config(model_name)
    config["num_classes"] = num_classes
    config["image_size"] = [img_size, img_size]
    return config


# export main model
def export_effdet_main(model_config: Dict, torch_model_path: str, out_folder: str, **model_kwargs):
    r'''
    model_path = os.path.join("~/project/ocr/models", "detection2023_custom.pth")
    '''
    model = EfficientDet(model_config, pretrained_backbone=True, **model_kwargs)  # class_out, box_out = model(x)
    model.class_net = HeadNet(model_config, num_outputs=model_config.num_classes)

    checkpoint = torch.load(torch_model_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        try:
            model.load_state_dict(checkpoint)
        except:
            raise ValueError(f"cannot load model state from {torch_model_path}")
    model.eval()

    # Export the model
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    output_fpath = os.path.join(out_folder, "effdet_main.onnx")

    img_size = tuple(model_config.image_size)
    sample_input = torch.rand((1, 3, *img_size))
    output_names = [
        *(f"cls_output{i}" for i in range(model_config['num_levels'])),
        *(f"box_output{i}" for i in range(model_config['num_levels'])),
    ]

    try:
        torch.onnx.export(
            model,
            sample_input, # model input (or a tuple for multiple inputs)
            output_fpath,
            export_params=True, # store the trained parameter weights inside the model file
            do_constant_folding=True,
            opset_version=12, # Operator support version
            input_names=['input'],
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch'},
                **{k: {0: 'batch'} for k in output_names},
            },
            verbose=False,
        )
        return 0
    except:
        return -1


# export effdet post process
def export_effdet_post_process(model_config: Dict, out_folder: str):
    r"""
    if num_levels == 5 and image_size == [512, 512] and num_classes == 1:
        dummy_cls_input_size = [
            (1, 9, 64, 64),
            (1, 9, 32, 32),
            (1, 9, 16, 16),
            (1, 9, 8, 8),
            (1, 9, 4, 4),
        ]
        dummpy_box_input_size = [
            (1, 36, 64, 64),
            (1, 36, 32, 32),
            (1, 36, 16, 16),
            (1, 36, 8, 8),
            (1, 36, 4, 4),
        ]
    """
    from effdet_convert.post_process import EffdetPostProcess

    model = EffdetPostProcess(model_config.num_levels, model_config.num_classes)
    model.eval()

    # dummy input
    level_ch = int(9 * model_config.num_classes)
    img_size = np.array(model_config.image_size)
    level_size = []
    for i in range(model_config.num_levels):
        _size = (img_size / (2**(i+3))).astype(int)
        level_size.append(tuple(_size))

    cls_input = [torch.rand((1, level_ch, *feature_size)) for feature_size in level_size]
    box_input = [torch.rand((1, 36, *feature_size)) for feature_size in level_size]

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    output_fpath = os.path.join(out_folder, "effdet_post.onnx")

    input_names = [
        *(f"cls_input{i}" for i in range(model_config['num_levels'])),
        *(f"box_input{i}" for i in range(model_config['num_levels'])),
    ]

    try:
        torch.onnx.export(
            model,
            (cls_input, box_input),
            output_fpath,
            export_params=True,
            do_constant_folding=True,
            opset_version=12,
            input_names=input_names,
            output_names=['cls_outputs', 'box_outputs', 'indices', 'classes'],
            dynamic_axes={
                **{k: {0: 'batch'} for k in input_names},
                'cls_outputs': {0: 'batch'},
                'box_outputs': {0: 'batch'},
                'indices': {0: 'batch'},
                'classes': {0: 'batch'},
            },
            verbose=False,
        )
        return 0
    except:
        return -1


# export effdet nms
def export_effdet_nms(model_config: Dict, out_folder: str, max_det_per_image: int = 100):
    from effdet_convert.nms import EffdetNMS

    model = EffdetNMS(model_config, max_det_per_image=max_det_per_image)
    model.eval()

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    output_fpath = os.path.join(out_folder, "effdet_nms.onnx")

    dummy_input = (
        torch.rand((1, 5000, 1)),
        torch.rand((1, 5000, 4)),
        torch.randint(5000, size=(1, 5000)),
        torch.zeros((1, 5000)),
    )

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_fpath,
            export_params=True,
            do_constant_folding=True,
            opset_version=12,
            input_names=['cls_outputs', 'box_outputs', 'indices', 'classes'],
            output_names=['output'],
            dynamic_axes={
                'cls_outputs': {0: 'batch'},
                'box_outputs': {0: 'batch'},
                'indices': {0: 'batch'},
                'classes': {0: 'batch'},
                'output': {0: 'batch'},
            },
            verbose=False,
        )
        return 0
    except:
        return -1
