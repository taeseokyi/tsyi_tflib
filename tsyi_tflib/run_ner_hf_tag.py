from __future__ import absolute_import, division, print_function

import argparse
from email import contentmanager
from inspect import isdatadescriptor
import json
import logging
import math
import os
import re
import unicodedata

import numpy as np
from regex import F
import tensorflow as tf
from fastprogress import master_bar, progress_bar
from seqeval.metrics import classification_report

# pylint: disable=no-member
# BertNer_ORG 는 공개 학습 bert 모델이 tf 2.x 와 호환되는 것이 없다.
# pylint:disable=unused-import
from transformers import AutoTokenizer, PreTrainedTokenizerFast, TFBartModel, TFElectraModel
from model import BertNerHF_tag

from optimization import AdamWeightDecay, WarmUp
from tokenization import HubPreprocessor, FullTokenizer_mecab as FullTokenizer, preprocess_text, convert_to_unicode, SPIECE_UNDERLINE
from official.common import distribute_utils

# https urlopen 오류 대응: export SSL_CERT_DIR=/etc/ssl/certs 를 해주거나, 아래를 실행함.
'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
'''  # pylint:disable=pointless-string-statement

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, input_txt=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.valid_ids = valid_ids
    self.label_mask = label_mask
    self.input_txt = input_txt


def readfile(filename):
  '''
  read file
  '''
  f = open(filename)
  data = []

  isdec = False
  okfirst = False
  for line in f:
    if not line.startswith('##'):
      continue
    else:
      if line[2:-1].isdecimal():
        isdec = True
        okfirst = False
        continue
      if isdec:
        atext = line[2:-1]
        isdec = False
        okfirst = True
        continue
      if okfirst:
        btext = line[2:-1]
        data.append((atext, btext))
        okfirst = False

  return data


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(self, input_file, quotechar=None):
    """Reads a tab separated value file."""
    return readfile(input_file)


class NerProcessor(DataProcessor):
  """Processor for the CoNLL-2003 data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    #return ["O", "B-TERM", "I-TERM", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    labels = ["O"]
    labels += ["B-AFW", "B-PER", "B-LOC", "B-ORG", "B-NUM", "B-ANM", "B-CVL", "B-DAT", "B-TRM", "B-FLD", 'B-TIM', "B-EVT", "B-PLT", "B-MAT"]
    labels += ["I-AFW", "I-PER", "I-LOC", "I-ORG", "I-NUM", "I-ANM", "I-CVL", "I-DAT", "I-TRM", "I-FLD", 'I-TIM', "I-EVT", "I-PLT", "I-MAT"]
    labels += ["[CLS]", "[SEP]"]  # [SEP] 은 끝나는 조건으로 항상 마지막에 위치
    return labels

  def _create_examples(self, lines, set_type):
    examples = []
    for i, (sentence, label) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      examples.append(InputExample(
          guid=guid, text_a=sentence, text_b=label, label=None))
    return examples


org_txt = '대북(對北) 인권단체인 ‘북한인권개선모임’ 등에 따르면 북한 노동자 6만∼7만 명이 러시아 중동 아프리카 중국 등에서 일하면서 연간 수억 달러 이상 벌어들인다.'
tag_txt = '대북(對北) 인권단체인 ‘<북한인권개선모임:ORG>’ 등에 따르면 <북한:LOC> <노동자:CVL> <6만∼7만 명:NUM>이 <러시아 중동 아프리카 중국:LOC> 등에서 일하면서 <연간:DAT> 수억 <달러:CVL> 이상 벌어들인다.'
p1 = re.compile(r'<\s*[^:]+:\w+\s*>')
m = re.findall(p1, tag_txt) # 태깅 영역 모두 찿아줌.
p2 = re.compile(r'<\s*([^:]+):(\w+)\s*>')
m = p2.match('<필리핀:LOC>') # m.groups()  --> ('필리핀', 'LOC')

def cov_tag_text(org_txt, tag_txt, tokenizer):
  tok = tokenizer.tokenize(org_txt)
  _txt = p1.sub('▁', tag_txt)
  tags = [ p2.match(x).groups() for x in p1.findall(tag_txt)]
  tag_index = 0
  label = []
  _index = 0
  len_txt = len(_txt)
  len_tag = len(tags)
  word = ""
  for token in tok:
    if token[0] in ('▁', '#'): # sp_model 단어 시작표시
      tt = token[1:]
    else:
      tt = token
    ch = _txt[_index]
    while ch == ' ':
      if _index < len_txt:
        _index += 1
      else:
        break
      ch = _txt[_index]
    if ch == '▁':
      if word == '':
        (word, tag) = tags[tag_index]  # 태깅된 단어와 태그
        word = word.replace(' ', '')
        prefix_tag = 'B-'
      else:
        prefix_tag = 'I-'
      if len(tt) > len(word) and word == tt[:len(word)]:
        if _index < len_txt:
          _index += 1
        else:
          break
        if tt[len(word):] == _txt[_index:_index + (len(tt) - len(word))]:  # 마지막 조사까지 같은지 토큰 전체 비교
          label.append(f'{prefix_tag}{tag}')
          if tag_index < len_tag:
            tag_index += 1
          else:
            break
          if _index + (len(tt) - len(word)) < len_txt:
            _index += len(tt) - len(word)
            word = ""
            continue
          else:
            break
        else:
          break
      if tt == word[:len(tt)]:
        label.append(f'{prefix_tag}{tag}')
        word = word[len(tt):]
        if word == '':
          if tag_index < len_tag:
            tag_index += 1
          else:
            break
          if _index < len_txt:
            _index += 1
          else:
            break
        continue
    
    if tt == _txt[_index : _index + len(tt)]:
      label.append('O')
      if _index + len(tt) < len_txt:
        _index += len(tt)
      else:
        break
    else:
      break
  print(tok)
  print(label)
  assert(len(tok) == len(label))
  return (tok, label)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, is_spm_model=False, do_lower_case=False):
  """Loads a data file into a list of `InputBatch`s.  KoElectra is wordpiece. Therfore is_spm_mode is Fasle."""
  
  label_map = {label: i for i, label in enumerate(label_list, 1)}
  
  features = []
  for (ex_index, example) in enumerate(examples):
    if is_spm_model:
      line_a = preprocess_text(example.text_a, lower=do_lower_case)
      line_b = preprocess_text(example.text_b, lower=do_lower_case)
    else:
      line_a = convert_to_unicode(example.text_a)
      line_b = convert_to_unicode(example.text_b)

    line_a = unicodedata.normalize('NFC', line_a)
    line_b = unicodedata.normalize('NFC', line_b)
    (_, labellist) = cov_tag_text(line_a, line_b, tokenizer)
    
    label_ids = []
    label_ids.append(label_map["[CLS]"])  # 첫 토큰 추가
    for label in labellist: # 라벨 토큰 변환
      label_ids.append(label_map[label])
    label_ids.append(label_map["[SEP]"])
    
    valid = [1] * max_seq_length  # 입력 대응, [CLS], [SEP] 라벨 추가 토큰 대응 loss 계산용 valid 마스크 추가 (== 모두 1 ^^)
    label_mask = [True] * len(label_ids)  # 입력 대응, 가변 이력 유효 길이 계산용 마스크, 각 입력 True, 패딩은 Fase
    
    while len(label_ids) < max_seq_length:  # 여기서는 입력 토큰과 라벨 토큰의 갯수가 같다.
      label_ids.append(0) # 0 is label padding.
      label_mask.append(False)

    tokens = tokenizer(line_a, padding = 'max_length', max_length = max_seq_length)
    input_ids = tokens['input_ids']
    input_mask = tokens['attention_mask']
    segment_ids = tokens['token_type_ids']

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(valid) == max_seq_length
    assert len(label_mask) == max_seq_length

    if ex_index < 5:
      # pylint:disable=logging-not-lazy
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("labels: %s" % " ".join(
          [str(x) for x in labellist]))
      logger.info("label_mask: %s" %
                  " ".join([str(x) for x in label_mask]))
      logger.info("label_ids: %s" %
                  " ".join([label_list[x-1] for x in label_ids]))
      logger.info("input_txt: %s" % line_a)
      logger.info("valid: %s" % " ".join([str(x) for x in valid]))

    features.append(
        InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_ids,
                      valid_ids=valid,
                      label_mask=label_mask,
                      input_txt=line_a))
  return features


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--hf_model_name", default="monologg/koelectra-small-v2-discriminator", type=str, required=True,
                      help="String that represents tensorflow hugging face's model name.")
  parser.add_argument("--do_spm_model", action='store_true',
                      help="Tokenizer's type.")
  # Other parameters
  parser.add_argument("--max_seq_length",
                      default=128,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
  parser.add_argument("--do_train",
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval",
                      action='store_true',
                      help="Whether to run eval on the dev/test set.")
  parser.add_argument("--eval_on",
                      default="dev",
                      type=str,
                      help="Evaluation set, dev: Development, test: Test")
  parser.add_argument("--do_lower_case",
                      action='store_true',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument("--train_batch_size",
                      default=32,
                      type=int,
                      help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
                      default=64,
                      type=int,
                      help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
                      default=3,
                      type=int,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                           "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--weight_decay", default=0.01, type=float,
                      help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="random seed for initialization")
  # training stratergy arguments
  parser.add_argument("--multi_gpu",
                      action='store_true',
                      help="Set this flag to enable multi-gpu training using MirroredStrategy."
                           "Single gpu training")
  parser.add_argument("--gpus", default='0', type=str,
                      help="Comma separated list of gpus devices."
                      "For Single gpu pass the gpu id.Default '0' GPU"
                      "For Multi gpu,if gpus not specified all the available gpus will be used")
  parser.add_argument("--tpu", default=None, type=str,
                      help="Optional. String that represents TPU to connect to. Must not be None if `distribution_strategy` is set to `tpu`")

  parser.add_argument("--init_hf_model", default=None, type=str,
                      help="Optional. String that represents a pre-trained file in the hugging-face model directory for fine-tuning.")
  args = parser.parse_args()

  processor = NerProcessor()
  label_list = processor.get_labels()
  num_labels = len(label_list) + 1 # 0 is label padding. 따라서 num_labels 는 실제 갯수 보다 1 큼.

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError(
        "Output directory ({}) already exists and is not empty.".format(args.output_dir))
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  if args.do_train:
    # using hubhugging face's a transformer model name which is a pretrained language model.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.hf_model_name)

  if args.multi_gpu:
    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy="mirrored",
        num_gpus=len(args.gpus.split(',')))
  else:
    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy="one_device",
        num_gpus=1)

  train_examples = None
  optimizer = None
  num_train_optimization_steps = 0
  ner = None
  if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size) * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion *
                       num_train_optimization_steps)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate,
                                                                     decay_steps=num_train_optimization_steps, end_learning_rate=0.0)
    if warmup_steps:
      learning_rate_fn = WarmUp(initial_learning_rate=args.learning_rate,
                                decay_schedule_fn=learning_rate_fn,
                                warmup_steps=warmup_steps)
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=args.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=args.adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    with strategy.scope():
      ner = BertNerHF_tag(TFBartModel, tf.float32, num_labels, args.max_seq_length, args.hf_model_name, args.init_hf_model)
      loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)

  label_map = {i: label for i, label in enumerate(label_list, 1)}
  if args.do_train:
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, is_spm_model=args.do_spm_model, do_lower_case=args.do_lower_case)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in train_features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in train_features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in train_features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in train_features]))
    all_label_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_mask for f in train_features]))

    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in train_features]))
    all_input_txt = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_txt for f in train_features]))

    # Dataset using tf.data
    train_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_label_mask, all_input_txt))
    shuffled_train_data = train_data.shuffle(buffer_size=int(len(train_features) * 0.1),
                                             seed=args.seed,
                                             reshuffle_each_iteration=True)
    batched_train_data = shuffled_train_data.batch(args.train_batch_size)
    # Distributed dataset
    #dist_dataset = strategy.experimental_distribute_dataset(batched_train_data)
    dist_dataset = batched_train_data

    loss_metric = tf.keras.metrics.Mean()

    epoch_bar = master_bar(range(args.num_train_epochs))
    pb_max_len = math.ceil(
        float(len(train_features))/float(args.train_batch_size))

    def train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt):
      def step_fn(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt):

        with tf.GradientTape() as tape:
          logits = ner(input_ids, input_mask, segment_ids,
                       valid_ids, input_txt=input_txt, training=True)
          label_mask = tf.reshape(label_mask, (-1,))
          logits = tf.reshape(logits, (-1, num_labels))
          logits_masked = tf.boolean_mask(logits, label_mask)
          label_ids = tf.reshape(label_ids, (-1,))
          label_ids_masked = tf.boolean_mask(label_ids, label_mask)
          cross_entropy = loss_fct(label_ids_masked, logits_masked)
          loss = tf.reduce_sum(cross_entropy) * (1.0 / args.train_batch_size)
        grads = tape.gradient(loss, ner.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, ner.trainable_variables)))
        return cross_entropy

      per_example_losses = strategy.run(step_fn,
                                        args=(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt))
      mean_loss = strategy.reduce(
          tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
      return mean_loss

    for epoch in epoch_bar:  # pylint:disable=unused-variable
      with strategy.scope():
        for (input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask, input_txt) in progress_bar(dist_dataset, total=pb_max_len, parent=epoch_bar):
          loss = train_step(input_ids, input_mask, segment_ids,
                            valid_ids, label_ids, label_mask, input_txt)
          loss_metric(loss)
          epoch_bar.child.comment = f'loss : {loss_metric.result()}'
      loss_metric.reset_states()

    # model weight save
    ner.save_weights(os.path.join(args.output_dir, "model.h5"))

    # copy vocab to output_dir
    # copy bert config to output_dir

    ner.config.to_json_file(os.path.join(
        args.output_dir, 'transformer_encoder.config'))
    #tokenizer.save_vocabulary(args.output_dir)

    # save label_map and max_seq_length of trained model
    model_config = {"bert_model": args.hf_model_name, "do_lower": args.do_lower_case,
                    "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                    "label_map": label_map}
    json.dump(model_config, open(os.path.join(
        args.output_dir, "model_config.json"), "w"), indent=4)

  if args.do_eval:
    # load tokenizer
    # config is json object, not str.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.hf_model_name)

    ner = BertNerHF_tag(TFBartModel, tf.float32, num_labels, args.max_seq_length, args.hf_model_name)

    ids = tf.ones((1, args.max_seq_length), dtype=tf.int64)
    _ = ner(ids, ids, ids, ids, input_txt=tf.constant(
        ['']), training=False)  # 배치자료가 입력되는 것으로 값을 제공
    ner.load_weights(os.path.join(
        args.output_dir, "model.h5"))  # 학습결과 모델 새로 로딩

    # load test or development set based on argsK
    if args.eval_on == "dev":
      eval_examples = processor.get_dev_examples(args.data_dir)
    elif args.eval_on == "test":
      eval_examples = processor.get_test_examples(args.data_dir)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, is_spm_model=args.do_spm_model, do_lower_case=args.do_lower_case)
    logger.info("***** Running evalution *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in eval_features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in eval_features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in eval_features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in eval_features]))

    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in eval_features]))
    all_input_txt = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_txt for f in eval_features]))

    eval_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_input_txt))
    batched_eval_data = eval_data.batch(args.eval_batch_size)

    loss_metric = tf.keras.metrics.Mean()
    epoch_bar = master_bar(range(1))
    pb_max_len = math.ceil(
        float(len(eval_features))/float(args.eval_batch_size))

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for epoch in epoch_bar:
      for (input_ids, input_mask, segment_ids, valid_ids, label_ids, input_txt) in progress_bar(batched_eval_data, total=pb_max_len, parent=epoch_bar):
        logits = ner(input_ids, input_mask,
                     segment_ids, valid_ids, training=False, input_txt=input_txt)
        logits = tf.argmax(logits, axis=2)
        for i, label in enumerate(label_ids):
          temp_1 = []
          temp_2 = []
          for j, _ in enumerate(label):
            if j == 0:
              continue
            elif label_map[label_ids[i][j].numpy()] == '[SEP]':  # 끝나는 조건 [SEP] 이면 탈출
              y_true.append(temp_1)
              y_pred.append(temp_2)
              break
            else:
              temp_1.append(label_map[label_ids[i][j].numpy()])
              temp_2.append(label_map[logits[i][j].numpy()])
    report = classification_report(y_true, y_pred, digits=4)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      logger.info("***** Eval results *****")
      logger.info("\n%s", report)
      writer.write(report)


if __name__ == "__main__":
  main()

# pylint: disable=pointless-string-statement
'''

python run_ner_hf_tag.py  \
  --data_dir=data/tech_name_tag  \
  --hf_model_name=monologg/koelectra-small-v2-discriminator  \
  --output_dir=out_base_ner_hf  \
  --max_seq_length=512  \
  --do_train  \
  --num_train_epochs=3  \
  --do_eval  \
  --eval_on=dev  \
  --train_batch_size=4

  --multi_gpu \
  --gpus=0,1

python run_ner_hf_tag.py  \
  --data_dir=data/tech_name_tag  \
  --hf_model_name=gogamza/kobart-base-v2  \
  --do_spm_model  \
  --output_dir=out_base_ner_kobart  \
  --max_seq_length=512  \
  --do_train  \
  --num_train_epochs=3  \
  --do_eval  \
  --eval_on=dev  \
  --train_batch_size=2

'''
