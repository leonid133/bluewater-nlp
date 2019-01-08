import hydro_serving_grpc as hs
import numpy as np
from pymystem3 import Mystem
import re
import os

HS_MAPPING_PATH = '/model/files/ru-rnc.map'
mapping_path = os.environ.get('MAPPING_PATH', HS_MAPPING_PATH)
rnc2univ_mapping = {}
with open(mapping_path) as inp:
    map_txt = inp.read()
for pair in map_txt.split('\n'):
    pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
    if len(pair) > 1:
        rnc2univ_mapping[pair[0]] = pair[1]
print(rnc2univ_mapping)
mystem = Mystem()


def infer_str(msg):
    processed = mystem.analyze(msg)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            if pos in rnc2univ_mapping:
                tagged.append(lemma + '_' + rnc2univ_mapping[pos])  # здесь мы конвертируем тэги
            else:
                tagged.append(lemma + '_X')  # на случай, если попадется тэг, которого нет в маппинге
        except KeyError:
            continue  # знаки препинания
    return tagged


def infer(msg):
    # TODO только для этого numpy?
    msg_arr = np.array(msg.string_val)
    msg_str = msg_arr[0]
    tokens = infer_str(msg_str)
    y = hs.TensorProto(
        dtype=hs.DT_STRING,
        string_val=[t.encode('utf-8', 'ignore') for t in tokens],
        tensor_shape=hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=-1)]))

    # 3. Return the result
    return hs.PredictResponse(outputs={"preprocessed_msg": y})
