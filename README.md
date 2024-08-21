# Export EfficientDet ONNX

Convert the EfficientDet model from PyTorch to ONNX.

## Export EfficientDet ONNX model

```=shell
$ python ./main.py -n "tf_efficientdet_d0" -m "<pytorch model pth(.pth)>" -s <image size> -c <num classes> -o "<ONNX model folder>" --max_detect <max detection boxes>
```

## Inference with ONNX
Use `EffdetONNXInfer` to run Effdet inference.
