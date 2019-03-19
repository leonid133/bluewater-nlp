import hydro_serving_grpc as hs
import numpy as np
import logging
import os
from bluewater.nlp.preprocess import get_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
HS_MAPPING_PATH = '/model/files/ru-rnc.map'
mapping_path = os.environ.get('MAPPING_PATH', HS_MAPPING_PATH)
pipe = get_preprocessing_pipeline(mapping_path)


def infer_str(msg):
    cas = pipe.process_txt(msg)
    return cas.lemma_poses


def infer(msg):
    # TODO только для этого numpy?
    msg_arr = np.array(msg.string_val)
    msg_str = msg_arr[0]
    tokens = infer_str(msg_str)
    y = hs.TensorProto(
        dtype=hs.DT_STRING,
        string_val=[t.encode('utf-8', 'ignore') for t in tokens],
        tensor_shape=hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=len(tokens))]))

    # 3. Return the result
    return hs.PredictResponse(outputs={"preprocessed_msg": y})
