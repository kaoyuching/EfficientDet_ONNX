from typing import List, Dict, Tuple
import re
import numpy as np

import onnx
from onnx.mapping import TENSOR_TYPE_MAP
import onnxruntime as ort


class EffdetONNXInfer():
    _tensor_type_mapping = {v.name.split('.')[-1].lower(): v.np_dtype for k, v in TENSOR_TYPE_MAP.items()}

    def __init__(
        self,
        main_model_path,
        post_model_path,
        nms_model_path,
        device: int = -1
    ):
        self.device = device
        self._load_models(main_model_path, post_model_path, nms_model_path)

    def _load_models(self, main_model_path: str, post_model_path: str, nms_model_path: str):
        sess_options = ort.SessionOptions()
        if self.device >= 0:
            providers = [
               ('CUDAExecutionProvider', {'device_id': self.device}), 
               'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        self.main_session = ort.InferenceSession(main_model_path, sess_options=sess_options, provider=providers)
        self.post_session = ort.InferenceSession(post_model_path, sess_options=sess_options, provider=providers)
        self.nms_session = ort.InferenceSession(nms_model_path, sess_options=sess_options, provider=providers)

    def _tensor_dtype_to_np_dtype(self, type_name: str):
        r"""
        'tensor(float)' -> np.float64
        """
        str_type = re.search('(?<=tensor\()\w*', type_name).group(0)
        return self._tensor_type_mapping[str_type]

    def handle_data_io(self, session, data: List[np.ndarray]) -> Tuple[Dict, List]:
        r"""
        Return:
            - input_data: Dict
            - output_names: List
        """
        if not isinstance(data, list):
            data = [data]
        input_data = {
            node.name: x.astype(self._tensor_dtype_to_np_dtype(node.type)) 
            for node, x in zip(session.get_inputs(), data)
        }
        output_names = [node.name for node in session.get_outputs()]
        return input_data, output_names

    def __call__(self, img) -> np.ndarray:
        r"""
        Input:
            - img: shape (bs, c, h, w)

        Return:
            - shape (bs, num_boxes, 6)
        """
        main_input_data, main_output_names = self.handle_data_io(self.main_session, img)
        main_outputs = self.main_session.run(main_output_names, main_input_data)

        post_input_data, post_output_names = self.handle_data_io(self.post_session, main_outputs)
        post_outputs = self.post_session.run(post_output_names, post_input_data)

        nms_input_data, nms_output_names = self.handle_data_io(self.nms_session, post_outputs)
        nms_outputs = self.nms_session.run(nms_output_names, nms_input_data)
        return nms_outputs[0]
