import tensorflow as tf
import numpy as np
from pathlib import Path
import functools

import logging
import sys

import json
import argparse

import time

import os

# 只在CPU上运行的方法
os.environ["CUDA_DEVICE_ORDE"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--type', type=str, default='demo', help='extract/predict/demo/test')
parser.add_argument('--model_dir', type=str, default='model', help='model file')
parser.add_argument('--data_dir', type=str, default='data', help='train.words.txt train.tags.txt')
parser.add_argument('--extract_data_name', type=str, default='train,testa,testb', help='train,testa,testb......')
parser.add_argument('--result_path', type=str, default='result', help='predict result')
args = parser.parse_args()

DATADIR = args.data_dir                     # 'G:/test_data/NLP/rmrb2014/ready_data'
MODEL_PATH = args.model_dir                 # 'G:/test_data/NLP/rmrb2014/results/md'
RESULTS_DIR = args.result_path              # 'G:/test_data/NLP/rmrb2014/results/predict'

_extract_data_name = args.extract_data_name.strip().split(',')

# Params
params = {
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'epochs': 30,
    'batch_size': 32,
    'buffer': 15000,
    'lstm_size': 300,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz'))
}

# Logging
Path(RESULTS_DIR).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.WARNING)
handlers = [
    logging.FileHandler(RESULTS_DIR + '/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

mylogger = logging.getLogger('ner_predict')
mylogger.setLevel(logging.WARNING)
fh = logging.FileHandler(RESULTS_DIR + '/ner_predict.log')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s[%(lineno)d] %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
mylogger.addHandler(fh)
mylogger.addHandler(ch)

def __generator_data(words):
    word_list = [word.encode() for word in words]
    tag_list = ["O".encode() for word in words]
    dataset = ((word_list, len(word_list)),tag_list)

    yield dataset

def _input_fn(words, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(__generator_data, words),
        output_shapes=shapes, output_types=types)

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset

def model_fn(features, labels, mode, params):
    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.]*params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def _build_model():
    cfg = tf.estimator.RunConfig() # 可以不需要
    estimator = tf.estimator.Estimator(model_fn, MODEL_PATH, cfg, params)
    return estimator

class IdentityNER(object):

    def __init__(self):
        self._tag_pred_b = ""
        self._word_index_b = -1
        self._tag_pred_i = ""
        self._pred_word =""
        #self.tag_words = {}
        self.tag_words = []


    def _set_tag_words_for_dict(self):
        _key = self._tag_pred_b[2:]
        if _key in self.tag_words.keys():
            _item = self.tag_words[_key]
            _item_key = self._pred_word
            if _item_key in _item.keys():
                _item[_item_key] += 1
            else:
                _item[_item_key] = 1
        else:
            _item = {self._pred_word: 1}
            self.tag_words[_key] = _item

    def _set_tag_words_for_list(self):
        # [{"word":"佛山","index":0,"ner":"LOC"},{"word":"佛山","index":5,"ner":"LOC"}]
        _ner_type = self._tag_pred_b[2:]
        _ner_item = {"word": self._pred_word, "index": self._word_index_b, "ner": _ner_type}
        self.tag_words.append(_ner_item)

    def get_pred_result(self):
        return self.tag_words

    def _append_tag_word(self):
        if len(self._tag_pred_b) > 0:  # 之前存在，需要保存
            self._set_tag_words_for_list()
            return True
        else:
            return False

    def process_sent(self, words, tags, preds_tages):
        self._tag_pred_b = ""
        self._tag_pred_i = ""
        self._pred_word = ""
        self._word_index_b = -1

        word_index = 0
        for word, tag, tag_pred in zip(words, tags, preds_tages):
            self._process_word(word, tag_pred, word_index)
            word_index += 1

        if len(self._tag_pred_b) > 0:
            self._append_tag_word()
            self._tag_pred_b = ""
            self._pred_word = ""
            self._word_index_b = -1

    def _process_word(self, word, tag_pred, word_index):
        _word = word.decode()
        _tag_pred = tag_pred.decode()
        if _tag_pred != 'O':
            if _tag_pred[0] == 'B':  # 新词开始
                self._append_tag_word()
                self._tag_pred_b = _tag_pred
                self._pred_word = _word
                self._word_index_b = word_index
            elif _tag_pred[0] == 'I':  # 词中间
                if len(self._tag_pred_b) > 0 and self._tag_pred_b[2:] == _tag_pred[2:]:
                    self._pred_word += _word
                else:
                    assert (True)
        else:
            _is_append = self._append_tag_word()
            if _is_append:
                self._tag_pred_b = ""
                self._pred_word = ""
                self._word_index_b = -1

    def process(self, golds_gen, preds_gen):
        i = 0
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), tags) = golds
            self.process_sent(words, tags, preds['tags'])

            i += 1
            if i % 1000 == 0:
                print("process....:{}".format(i))

        return self.tag_words

def __predict(sent, estimator, _tag_words):
    test_inpf = functools.partial(_input_fn, sent)
    golds_gen = __generator_data(sent)
    preds_gen = estimator.predict(test_inpf)

    def __append_tag_word():
        if len(_tag_pred_b) > 0:  # 之前存在，需要保存
            _key = _tag_pred_b[2:]
            if _key in _tag_words.keys():
                _tag_words[_key].append(_pred_word)
            else:
                _tag_words[_key] = [_pred_word]

            return True
        else:
            return False

    def _append_tag_word_dict():
        if len(_tag_pred_b) > 0:  # 之前存在，需要保存
            _key = _tag_pred_b[2:]
            if _key in _tag_words.keys():
                _item = _tag_words[_key]
                _item_key = _pred_word
                if _item_key in _item.keys():
                    _item[_item_key] += 1
                else:
                    _item[_item_key] = 1
            else:
                _item = {_pred_word: 1}
                _tag_words[_key] = _item

            return True
        else:
            return False

    for golds, preds in zip(golds_gen, preds_gen):
        _tag_pred_b = ""
        _tag_pred_i = ""
        _pred_word =""
        ((words, _), tags) = golds
        for word, tag, tag_pred in zip(words, tags, preds['tags']):
            _word = word.decode()
            _tag_pred = tag_pred.decode()
            if _tag_pred != 'O':
                if _tag_pred[0] == 'B': # 新词开始
                    _append_tag_word_dict()
                    _tag_pred_b = _tag_pred
                    _pred_word = _word
                elif _tag_pred[0] == 'I': # 词中间
                    if len(_tag_pred_b) > 0 and _tag_pred_b[2:] == _tag_pred[2:]:
                        _pred_word += _word
                    else:
                        assert (True)
            else:
                _is_append = _append_tag_word_dict()
                if _is_append:
                    _tag_pred_b = ""
                    _pred_word = ""

        if len(_tag_pred_b) > 0:
            _append_tag_word_dict()
            _tag_pred_b = ""
            _pred_word = ""

class BatchInput:

    def _fwords(self, name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def _ftags(self, name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    def _parse_fn(self, line_words, line_tags):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags), "Words and tags lengths don't match"
        return (words, len(words)), tags

    def generator_fn(self, words, tags):
        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield self._parse_fn(line_words, line_tags)

    def input_fn(self, words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')

        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset



# 外部调用接口(如：java)
def predict(sent):
    demo_sent = sent.strip()
    test_inpf = functools.partial(_input_fn, demo_sent)
    golds_gen = __generator_data(demo_sent)
    preds_gen = g_estimator.predict(test_inpf)
    _tag_words = IdentityNER().process(golds_gen, preds_gen)
    result_dict = {"sent": demo_sent, "result": _tag_words}
    json_str = json.dumps(result_dict, ensure_ascii=False)
    return json_str

# 初始化
mylogger.info("Begin Initialization.......")
g_estimator = _build_model()
# 初始化，第一次运行预测速度会很慢
json_str = predict("佛山天翼公司欢迎您！")
print(json_str)
mylogger.info("End Initialization.......")

