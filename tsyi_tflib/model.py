from __future__ import absolute_import, division, print_function

import tensorflow as tf

from transformers import PreTrainedTokenizerFast  # Registers the ops. pylint:disable=unused-import


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
