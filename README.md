# Export EfficientDet ONNX

Convert the EfficientDet model from PyTorch to ONNX.

## Export EfficientDet ONNX model

```=shell
$ python ./main_export.py -n "tf_efficientdet_d0" -m "<pytorch model pth(.pth)>" -s <image size> -c <num classes> -o "<ONNX model folder>" --max_detect <max detection boxes>
```

The efficientdet model is splited into three parts: main model, post process, and nms.   
The exported ONNX models are: `effdet_main.onnx`, `effdet_post.onnx`, and `effdet_nms.onnx`.


## Inference with ONNX
Use `EffdetONNXInfer` to run Effdet inference.

## Example
The example uses the effdet to detect car number plate. The torch model is fine-tuned with opendata using package `effdet`, which can be downloaded [here](https://huggingface.co/doriskao/effdet_d0_car_number_plate).

1. First, convert the torch model to onnx models.

```=shell
$ python ./main_export.py -n "tf_efficientdet_d0" -m "<model folder>/effdet_d0_car_number_plate.pth" -s 512 -c 1 -o "./onnx_models" --max_detect 10
```

After successfully converting models, there will be three models under the folder `onnx_models`: `effdet_main.onnx`, `effdet_post.onnx`, `effdet_nms.onnx`

2. Second, use `main_infer.py` to run the example.

```=shell
$ python ./main_infer.py
```


## Reference
- Effdet official repository: https://github.com/rwightman/efficientdet-pytorch/tree/master
