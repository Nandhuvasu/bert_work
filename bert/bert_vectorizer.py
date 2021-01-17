import os
import json
import time
import tarfile
import requests
import falcon
import tensorflow_text as text  # Registers the ops.
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model

from env_vars import constants
from bert.vectorizer_helper import reduce_mean

preprocessor_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2?tf-hub-format=compressed"
preprocessor_target_path = 'bert/models/preprocess/bert_multi_cased_preprocess.tar.gz'
bert_model_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3?tf-hub-format=compressed"
bert_model_target_path = 'bert/models/bert/bert_multi_cased_L-12_H-768_A-12.tar.gz'
LOCAL_BERT_DIR = 'bert/models/bert'
LOCAL_PREPROCESS_DIR = 'bert/models/preprocess'
BERT_MODEL_NAME = 'bert_multi_cased_L-12_H-768_A-12'
PREPROCESS_MODEL_NAME = 'bert_multi_cased_preprocess'
MODEL_CACHE = None

def download_model(url, target_path):
    response = requests.get(url, stream = True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    else:
        print('Error downloading preprocess model')
    print('#### PREPROCESSOR DOWNLOADED ###')

def extract_model_files():
    print('Extracting files...')
    extract_files(bert_model_target_path, LOCAL_BERT_DIR, BERT_MODEL_NAME)
    extract_files(preprocessor_target_path, LOCAL_PREPROCESS_DIR, PREPROCESS_MODEL_NAME)
    
def extract_files(target_path, local_dir, model_name):
    tar = tarfile.open(target_path, 'r:gz')
    tar.extractall(local_dir)
    tar.close()
    print('{0} extraction done!'.format(model_name))

# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer(
#         "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2")
# # preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2")
# print('#### PREPROCESSOR DOWNLOADED ###')
# encoder_inputs = preprocessor(text_input)  # dict with keys: 'input_mask', 'input_type_ids', 'input_word_ids'
# encoder = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
#     trainable=True)
# # encoder = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3")
# print('#### ENCODER DOWNLOADED ###')
# outputs = encoder(encoder_inputs)
# pooled_output = outputs["pooled_output"]      # [batch_size, 768].
# sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
# encoder_outputs = outputs["encoder_outputs"][11]
# model = Model(inputs=[text_input], outputs=[pooled_output, sequence_output, encoder_outputs])

def get_bert_and_preprocess_models():
    print('Checking if {0} exists'.format(bert_model_target_path))
    if os.path.isfile(bert_model_target_path):
        print('Bert model Found')
    else:
        print('Bert model Not found, downloading')
        download_model(bert_model_url, bert_model_target_path)
    print('Checking if {0} exists'.format(preprocessor_target_path))
    if os.path.isfile(preprocessor_target_path):
        print('Preprocess model Found')
        return
    else:
        print('Preprocess model Not found')
        download_model(preprocessor_url, preprocessor_target_path)

def load_model_into_cache():
    try:
        print('Using tensorflow to load model into cache...')

        global MODEL_CACHE
        if MODEL_CACHE is None:
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
            preprocessor = hub.KerasLayer(LOCAL_PREPROCESS_DIR)
            encoder_inputs = preprocessor(text_input)  # dict with keys: 'input_mask', 'input_type_ids', 'input_word_ids'
            encoder = hub.KerasLayer(LOCAL_BERT_DIR)
            outputs = encoder(encoder_inputs)
            pooled_output = outputs["pooled_output"]      # [batch_size, 768].
            sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
            encoder_outputs_layer_12 = outputs["encoder_outputs"][11]
            encoder_outputs_layer_11 = outputs["encoder_outputs"][10]
            # encoder_outputs_layer_10 = outputs["encoder_outputs"][9]
            # encoder_outputs_layer_9 = outputs["encoder_outputs"][8]
            MODEL_CACHE = Model(inputs=[text_input], outputs=[pooled_output, sequence_output, encoder_outputs_layer_12, encoder_outputs_layer_11])

        print('Ready to process vectors...')
    except Exception:
        print('Failed loading the model. Probably missing from the models directory.')
        raise


def download_and_load_model():
    try:
        get_bert_and_preprocess_models()
        extract_model_files()
        load_model_into_cache()
        # get_vector()
    except Exception as e:
        print('Downloading bert/extracting/loading failed!')
        print(e)

# DO AN AVERAGE OF ALL THE VECTORS (IN ENCODER OUTPUTS) GET A SINGLE 768 VECTOR OUTPUT - SEE IF IT MATCHES POOLED OUTPUT
def get_vector(payload, layer_number):
    vector = []
    print('$$$$ predict invoked $$$$')
    # print(encoder_inputs)
    if 'text' in payload and type(payload['text']) is list:
        pool_embs, sequence_embs, encoder_ops_12, encoder_ops_11 = MODEL_CACHE.predict(payload['text'])

    vector = pool_embs

    if layer_number is not None:
        if layer_number is 11:
            vector = reduce_mean(encoder_ops_11)
        elif layer_number is 12:
            vector = reduce_mean(encoder_ops_12)

    print('### PREDICT CALLED ###')
    # print(sequence_embs)
    # # print(tf.shape(pool_embs))
    # print(all_embs)
    # print(tf.shape(encoder_ops))
    return vector.tolist()

class BertVectorizer(object):
    def on_post(self, req, resp):
        print('POST request received')
        bert_vector_compute_time_start = time.time()
        # Process stream to json
        raw_json = req.bounded_stream.read()
        layer_number = None

        # Body validation
        if not raw_json:
            resp.status = falcon.HTTP_400
            return resp

        if 'layer' in req.params:
            layer_number = int(req.params['layer'])

        # Transform and get vert vectors
        payload = json.loads(raw_json, encoding='utf-8')
        bert_vectors = get_vector(payload, layer_number)

        # Could vectors be computed validation
        if len(bert_vectors) == 0:
            resp.status = falcon.HTTP_400
            return resp

        # Construct response payload
        result = {
            'vectors': bert_vectors,
            'model_version': constants['BERT_MODEL_NAME'],
            'took': '{number}ms'.format(number=(int(round((time.time() - bert_vector_compute_time_start) * 1000))))
        }

        resp.body = json.dumps(result)
        return resp

if __name__ == '__main__':
    download_and_load_model()
