import os

def convert_str_to_bool(env_var_str):
    if env_var_str is None or len(env_var_str) == 0:
        return None

    if env_var_str in ['True', 'true']:
        return True

    if env_var_str in ['False', 'false']:
        return False

constants = {
    'BERT_MODEL_NAME': os.getenv('BERT_MODEL_NAME') or 'bert_multi_cased_L-12_H-768_A-12',
    'LOCAL_BERT_DIR': 'bert/models/bert',
    'MODEL_SERVER_TIMEOUT': os.getenv('MODEL_SERVER_TIMEOUT') or '400',
    'MODEL_SERVER_WORKERS': os.getenv('MODEL_SERVER_WORKERS') or '1',
    'LOG_LEVEL': os.getenv('LOG_LEVEL') or 'INFO',
    'LOG_COMPUTATION_TIME': convert_str_to_bool(os.getenv('LOG_COMPUTATION_TIME')) or True,
    'TENSOR_COMPUTE_DELAY': os.getenv('TENSOR_COMPUTE_DELAY') or '900',
    'POLL_DELAY': os.getenv('POLL_DELAY') or '5',
    'PROCESS_DELAY': os.getenv('PROCESS_DELAY') or '0.01',
    'TENSOR_LOOP_DELAY': os.getenv('TENSOR_LOOP_DELAY') or '1',
    'INITIAL_STARTUP_DELAY': os.getenv('INITIAL_STARTUP_DELAY') or '0.0005',
    'KAFKA_URL': os.getenv('KAFKA_URL') or 'localhost:9092,localhost:9092',
    'TOPIC_STOCK': os.getenv('TOPIC_STOCK') or 'local-stock-content-vectors-1',
    'TOPIC_CUSTOM': os.getenv('TOPIC_CUSTOM') or 'local-custom-content-vectors-1',
    'IS_ACCESS_LOGGING_ENABLED': os.getenv('IS_ACCESS_LOGGING_ENABLED') or True,
    'STATS_LOOP_DELAY': os.getenv('STATS_LOOP_DELAY') or '0.5',
    'READINESS_PROBE_TENSOR_DELAY': os.getenv('READINESS_PROBE_TENSOR_DELAY') or '11',
    'EXTEND_TENSOR_COMPUTE_DELAY': os.getenv('EXTEND_TENSOR_COMPUTE_DELAY') or '9'
}