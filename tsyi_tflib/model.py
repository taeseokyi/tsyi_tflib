from __future__ import absolute_import, division, print_function

import os
import hashlib
import tempfile

import tensorflow as tf
import tensorflow_text as text  # Registers the ops. pylint:disable=unused-import
import tensorflow_hub as hub

from transformers import AutoTokenizer, PreTrainedTokenizerFast, TFBartModel, TFElectraModel  # Registers the ops. pylint:disable=unused-import
from bert_modeling import BertConfig, BertModel

# BertNer2가 get_transformer_encoder() 함수  사용
from official.nlp.bert.bert_models import get_transformer_encoder
from official.nlp.bert import configs as bert_configs
from official.nlp.albert import configs as albert_configs

# pylint:disable=unused-argument, unidiomatic-typecheck,arguments-differ,abstract-method


class BertNer_ORG(tf.keras.Model):

  def __init__(self, bert_model, float_type, num_labels, max_seq_length, final_layer_initializer=None):
    '''
    bert_model : string or dict
                 string: bert pretrained model directory with bert_config.json and bert_model.ckpt
                 dict: bert model config , pretrained weights are not restored
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNer_ORG, self).__init__()

    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    if type(bert_model) == str:
      bert_config = BertConfig.from_json_file(
          os.path.join(bert_model, "bert_config.json"))
    elif type(bert_model) == dict:
      bert_config = BertConfig.from_dict(bert_model)

    bert_layer = BertModel(config=bert_config, float_type=float_type)

    _, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)
    self.bert = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids], outputs=[sequence_output])
    if type(bert_model) == str:
      init_checkpoint = os.path.join(bert_model, "bert_model.ckpt")
      checkpoint = tf.train.Checkpoint(model=self.bert)
      checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range)
    self.dropout = tf.keras.layers.Dropout(
        rate=bert_config.hidden_dropout_prob)
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, **kwargs):
    sequence_output = self.bert(
        [input_word_ids, input_mask, input_type_ids], **kwargs)
    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits


# 주의) 토큰 처리기와 preprocessor에서의 토큰처리기가 완벽하게 일치하지 않으므로 단지, 텍스트 입력으로 임베딩결과만 쓸 경우에 HUB BERT + preprocessor를 쓴다.
#      label의 valid 정보를 만들기 위해 preprocessor 토큰 모델을 읽어서 처리해 보았으나, 매우 느린 단점이 있다.
#      또한 어절과 문장으로 처리 단위가 다르기 때문에 생성 토큰과 preprocessor 가 변환한 텍스트 토큰이 완전하게 일치할지 알 수 없음.
# HUB 모델 입력을 preprocessor를 쓸 경우 텍스트를 그대로 전달하여 모델에서 토큰을 자동으로 생성한다.
class BertNER_HubPreprocessor(tf.keras.Model):

  def __init__(self, float_type, num_labels, max_seq_length, final_layer_initializer=None,
               hub_module_url=None, preprocessor=None, hub_module_trainable=True, initializer_range=0.02, hidden_dropout_prob=0.1):
    '''
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNER_HubPreprocessor, self).__init__()
    assets_path = os.path.join(tempfile.gettempdir(), "tfhub_modules", hashlib.sha1(
        hub_module_url.encode("utf8")).hexdigest(), 'assets')
    vocab_file = os.listdir(assets_path)
    if len(vocab_file) > 0:
      self.vocab_file = os.path.join(assets_path, vocab_file[0])
      self.is_albert = not vocab_file[0] == 'vocab.txt'
    else:
      self.vocab_file = None
      self.is_albert = None

    # Step 1: tokenize batches of text inputs.
    # This SavedModel accepts up to 2 text inputs.
    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string)]
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(s) for s in text_inputs]
    # Step 2 (optional): modify tokenized inputs.
    pass  # pylint:disable=unnecessary-pass
    # Step 3: pack input sequences for the Transformer encoder.
    seq_length = max_seq_length  # Your choice here.
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    core_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
    outputs = core_model(encoder_inputs)
    self.bert = tf.keras.Model(text_inputs, outputs=outputs, name='core_model')

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.TruncatedNormal(
          stddev=initializer_range)  # initializer_range
    self.dropout = tf.keras.layers.Dropout(
        rate=hidden_dropout_prob)  # hidden_dropout_prob
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, input_txt=None, **kwargs):
    outputs = self.bert(input_txt, **kwargs)

    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs['sequence_output']

    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits


# tf2.x keras 형식 텐서플로우 공식 모델 BERT, ALBERT로 작업한다.
class BertNer(tf.keras.Model):

  def __init__(self, bert_model, bert_config_file, init_bert, float_type, num_labels, max_seq_length, final_layer_initializer=None):
    '''
    bert_model : string or dict
                 string: bert pretrained model directory with bert_config.json and bert_model.ckpt
                 dict: bert model config , pretrained weights are not restored
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNer, self).__init__()

    if type(bert_model) == str:
      listdir = os.listdir(bert_model)
      if bert_config_file == "bert_config.json" and bert_config_file in listdir:
        bert_config = bert_configs.BertConfig.from_json_file(
            os.path.join(bert_model, bert_config_file))
        self.is_albert = False
      elif bert_config_file in listdir:
        bert_config = albert_configs.AlbertConfig.from_json_file(
            os.path.join(bert_model, bert_config_file))
        self.is_albert = True
      else:
        raise ValueError("Not exist bert_config or albert_config")
    elif type(bert_model) == dict:
      if 'is_albert' in bert_model:
        del bert_model['is_albert']  # 추가한 정보 전달 원소 삭제
        bert_config = albert_configs.AlbertConfig.from_dict(bert_model)
        self.is_albert = True
      else:
        bert_config = bert_configs.BertConfig.from_dict(bert_model)
        self.is_albert = False

    transformer_encoder = get_transformer_encoder(bert_config, max_seq_length)
    self.bert = transformer_encoder  # This is bertmodel.
    if type(bert_model) == str:
      # 최초 시작 bert_model이며, do_eval에서 실행이 되지 않음. bert
      init_checkpoint = os.path.join(bert_model, init_bert)
      checkpoint = tf.train.Checkpoint(model=self.bert)
      if self.is_albert:
        # 1.x 변환 모델은 bias와 layer_norm 이 없기때문에 이를 무시한다.
        checkpoint.restore(init_checkpoint)
      else:
        # 1.x 변환 모델은 bias와 layer_norm 이 없기때문에 이를 무시한다.
        checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range)
    self.dropout = tf.keras.layers.Dropout(
        rate=bert_config.hidden_dropout_prob)
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, input_txt=None, **kwargs):
    # bert 출력 [sequence_output, cls_output(=pooled_output)]
    if self.is_albert and False:
      encoder_inputs = dict(
          input_word_ids=input_word_ids,
          input_mask=input_mask,
          input_type_ids=input_type_ids
      )
      outputs = self.bert(encoder_inputs, **kwargs)
    else:
      outputs = self.bert(
          [input_word_ids, input_mask, input_type_ids], **kwargs)

    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs['sequence_output']

    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits


# HUB 모델을 쓰면서 토큰나이저는 허브에 있는 vocab.txt 나 30k-clean.model을 사용하여 입력데이터를 만든다.
class BertNerHUB(tf.keras.Model):

  def __init__(self, float_type, num_labels, max_seq_length, final_layer_initializer=None,
               hub_module_url=None, hub_module_trainable=True, initializer_range=0.02, hidden_dropout_prob=0.1):
    '''
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNerHUB, self).__init__()

    # HUB 모델의 어휘 사전 정보를 읽는다. BERT는 vocab.txt 이고, ALBERT는 30k-clean.model 이다.
    assets_path = os.path.join(tempfile.gettempdir(), "tfhub_modules", hashlib.sha1(
        hub_module_url.encode("utf8")).hexdigest(), 'assets')
    vocab_file = os.listdir(assets_path)
    if len(vocab_file) > 0:
      self.vocab_file = os.path.join(assets_path, vocab_file[0])
      self.is_albert = not vocab_file[0] == 'vocab.txt'
    else:
      self.vocab_file = None
      self.is_albert = None

    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
    )
    core_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
    outputs = core_model(encoder_inputs)
    self.bert = tf.keras.Model(
        encoder_inputs, outputs=outputs, name='core_model')

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.TruncatedNormal(
          stddev=initializer_range)  # initializer_range
    self.dropout = tf.keras.layers.Dropout(
        rate=hidden_dropout_prob)  # hidden_dropout_prob
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, input_txt=None, **kwargs):
    encoder_inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids
    )
    outputs = self.bert(encoder_inputs, **kwargs)

    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs['sequence_output']

    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits


# tf2.x keras 형식 텐서플로우 작업을 위한 hugging face's pretrained model을 NER로 확장한다.
class BertNerHF(tf.keras.Model):

  def __init__(self,  hf_model, float_type, num_labels, max_seq_length, hf_model_name, final_layer_initializer=None, init_hf_model=None):
    '''
    bert_model : string or dict
                 string: bert pretrained model directory with bert_config.json and bert_model.ckpt
                 dict: bert model config , pretrained weights are not restored
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNerHF, self).__init__()

    print (hf_model_name)
    transformer_encoder = hf_model.from_pretrained(hf_model_name, from_pt=True)
    self.config = transformer_encoder.config

    encoder_inputs = dict(
        input_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        attention_mask=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        token_type_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
    )

    # transformer_encoder(encoder_inputs).keys()에서 선택: 'last_hidden_state' or 'encoder_last_hidden_state'
    outputs = transformer_encoder(encoder_inputs)['encoder_last_hidden_state']
    self.bert = tf.keras.Model(
        encoder_inputs, outputs=outputs, name='core_model')

    if init_hf_model is not None:
      # 최초 시작 bert_model이며, do_eval에서 실행이 되지 않음.
      self.bert = hf_model.from_pretrained(init_hf_model)

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.truncated_normal(stddev=0.02)  # bert_config.initializer_range = 0.02

    # bert_config.hidden_dropout_prob is 0.1
    self.dropout = tf.keras.layers.Dropout(
        rate=0.1)
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, input_txt=None, **kwargs):
    encoder_inputs = dict(
          input_ids=input_word_ids,
          attention_mask=input_mask,
          token_type_ids=input_type_ids
      )
    outputs = self.bert(encoder_inputs, **kwargs)

    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs #['last_hidden_state'] # shape=(2, 512, 256) last_hidden_state is KoElectra

    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits


class BertNerHF_tag(tf.keras.Model):

  def __init__(self,  hf_model, float_type, num_labels, max_seq_length, hf_model_name, final_layer_initializer=None, init_hf_model=None):
    '''
    bert_model : string or dict
                 string: bert pretrained model directory with bert_config.json and bert_model.ckpt
                 dict: bert model config , pretrained weights are not restored
    float_type : tf.float32
    num_labels : num of tags in NER task
    max_seq_length : max_seq_length of tokens
    final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
    '''
    super(BertNerHF_tag, self).__init__()

    print (hf_model_name)
    transformer_encoder = hf_model.from_pretrained(hf_model_name, from_pt=True)
    self.tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_model_name)
    self.config = transformer_encoder.config

    encoder_inputs = dict(
        input_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        attention_mask=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
        token_type_ids=tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32),
    )

    # transformer_encoder(encoder_inputs).keys()에서 선택: 'last_hidden_state' or 'encoder_last_hidden_state'
    outputs = transformer_encoder(encoder_inputs)['encoder_last_hidden_state']
    self.bert = tf.keras.Model(
        encoder_inputs, outputs=outputs, name='core_model')

    if init_hf_model is not None:
      # 최초 시작 bert_model이며, do_eval에서 실행이 되지 않음.
      self.bert = hf_model.from_pretrained(init_hf_model)

    if final_layer_initializer is not None:
      initializer = final_layer_initializer
    else:
      initializer = tf.keras.initializers.truncated_normal(stddev=0.02)  # bert_config.initializer_range = 0.02

    # bert_config.hidden_dropout_prob is 0.1
    self.dropout = tf.keras.layers.Dropout(
        rate=0.1)
    self.classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

  def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_mask=None, input_txt=None, **kwargs):
    encoder_inputs = dict(
          input_ids=input_word_ids,
          attention_mask=input_mask,
          token_type_ids=input_type_ids
      )
    outputs = self.bert(encoder_inputs, **kwargs)

    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs #['last_hidden_state'] # shape=(2, 512, 256) last_hidden_state is KoElectra

    valid_output = []
    for i in range(sequence_output.shape[0]):
      r = 0
      temp = []
      for j in range(sequence_output.shape[1]):
        if valid_mask[i][j] == 1:
          temp = temp + [sequence_output[i][j]]
        else:
          r += 1
      temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
      valid_output = valid_output + temp
    valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
    sequence_output = self.dropout(
        valid_output, training=kwargs.get('training', False))
    logits = self.classifier(sequence_output)
    return logits
