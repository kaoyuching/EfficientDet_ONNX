import os
import argparse

from effdet_convert.export import get_model_config, export_effdet_main, export_effdet_post_process, export_effdet_nms


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--model_name", type=str)
parser.add_argument("-m", "--torch_model_path", type=str)
parser.add_argument("-s", "--img_size", type=int, default=None)
parser.add_argument("-c", "--num_classes", type=int, default=None)
parser.add_argument("-o", "--out_folder", type=str, default="./onnx_models")
parser.add_argument("--max_detect", type=int, default=100)

args = parser.parse_args()

model_name = args.model_name
torch_model_path = args.torch_model_path
img_size = args.img_size
num_classes = args.num_classes
out_folder = args.out_folder
max_detect = args.max_detect

if not os.path.exists(torch_model_path):
    raise ValueError("model path do not exist.")


model_config = get_model_config(model_name, num_classes=num_classes, img_size=img_size)

export_main_status = export_effdet_main(model_config, torch_model_path, out_folder)
print('export main status:', export_main_status)
export_post_status = export_effdet_post_process(model_config, out_folder)
print('export post status:', export_post_status)
export_nms_status = export_effdet_nms(model_config, out_folder, max_det_per_image=max_detect)
print('export nms status:', export_nms_status)
