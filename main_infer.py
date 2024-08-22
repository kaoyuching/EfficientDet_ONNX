import numpy as np
import cv2

from effdet_convert.onnx_infer import EffdetONNXInfer
from effdet_convert.utils import decode_box


onnx_model = EffdetONNXInfer(
    "./onnx_models/effdet_main.onnx",
    "./onnx_models/effdet_post.onnx",
    "./onnx_models/effdet_nms.onnx",
)


img_size = onnx_model.image_size  #(h, w)
input_img = cv2.imread('./example_image/img1.png')
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
origin_h, origin_w = input_img.shape[:2]

img = cv2.resize(input_img, img_size, interpolation=cv2.INTER_LINEAR)  # shape (h, w, c)
img = np.transpose(img/255, (2, 0, 1)).astype(np.float32)
img = np.expand_dims(img, 0)  # shape (b, c, h, w)

outputs = onnx_model(img)  # shape (bs, max_num_boxes, 6)

# show the image
batch_size = outputs.shape[0]
threshold = 0.5 

for b in range(batch_size):
    output = outputs[b]
    res = output[output[:, 4] > threshold]  # shape (num_boxes, 6)

    boxes = res[:, :4]  # (xmin, ymin, xmax, ymax)
    for box in boxes:
        box = decode_box(*box, img_size, origin_w, origin_h)
        cv2.rectangle(input_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
    cv2.imwrite('./example_image/img1_res.png', cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
print("Finish to infer './example_image/img1.png', the result is saved in './example_image/img1_res.png'")
