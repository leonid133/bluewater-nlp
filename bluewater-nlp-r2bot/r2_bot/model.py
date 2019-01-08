from .train_io import BatchedDatasetRecord, DatasetRecord
from .common_utils import get_lines
import tensorflow as tf
from .api import *
from functools import partial
import numpy as np

def _batch_matmul(batch, weights):
    input_ndims = batch.get_shape().ndims
    if input_ndims == 2:
        return tf.matmul(batch, weights)
    if input_ndims != 3:
        raise ValueError("_batch_matmul works only with 2-3 ranked input tensors, but got: " + str(batch))
    input_shape = tf.shape(batch)
    batch_reshaped = tf.reshape(batch, (-1, input_shape[2]))
    result = tf.matmul(batch_reshaped, weights)
    return tf.reshape(result, (input_shape[0], input_shape[1], weights.shape[1]))


def linear(input_batch, output_size, name, initializer=None, init_bias=0.0):
    input_size = input_batch.shape.as_list()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("weight_matrix",
                            [input_size, output_size],
                            tf.float32,
                            initializer=initializer)
    if init_bias is None:
        return _batch_matmul(input_batch, W), W
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return _batch_matmul(input_batch, W) + b, W, b


def model_fn(features, labels, mode, params):
    if isinstance(features, (BatchedDatasetRecord, DatasetRecord)):
        features = features._asdict()
    tokens = features['tokens']
    token_lengths = features['tokens_len']
    #
    vocab_fp = params['vocab_filepath']
    word_emb_len = get_lines(vocab_fp)
    oov_word_i = word_emb_len - 1
    word_i_lookup = tf.contrib.lookup.index_table_from_file(vocab_fp, 
                                                            default_value=oov_word_i,
                                                            name='word_i_lookup_table')
    #
    word_indices = word_i_lookup.lookup(tokens, name='word_indices')
    #
    emb_params = tf.get_variable('word_embeddings', 
                                 shape=(word_emb_len, params['pretrained.dim']),
                                 dtype=params['pretrained.dtype'])
    word_vecs = tf.nn.embedding_lookup(emb_params, word_indices)
      
    # ENCODER LSTM RNN
    encoder_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(params['encoder.hidden_size'])
    _, encoder_state_tuple = tf.nn.dynamic_rnn(encoder_rnn_cell, word_vecs, token_lengths,
                                               dtype=tf.float32, scope="encoder_rnn")
    encoder_state = encoder_state_tuple.c
    # LAYER softmax
    # shape (batch, label_num)
    logits, out_W, out_b = linear(encoder_state, NUM_INTENTS, "label_logits")
    # probabilies, shape (batch, max_time, label_num)
    intent_proba = tf.nn.softmax(logits, name='intent_proba')
    intent_confidence = tf.reduce_max(intent_proba, axis=-1)
    predicted_intent_i = tf.argmax(logits, -1, name='predicted_intent_i')
    label_lookup = tf.contrib.lookup.index_to_string_table_from_tensor(
        INTENT_LABELS, 
        default_value=INTENT_LABELS[-1],
        name='label_lookup_table')
    predicted_intent = label_lookup.lookup(predicted_intent_i, name='predicted_intent')
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={
            'intent_i' : predicted_intent_i,
            'confidence' : intent_confidence,
            'input_length' : token_lengths,
            'intent' : predicted_intent
        })
    
    label_reverse_lookup = tf.contrib.lookup.index_table_from_tensor(INTENT_LABELS, 
                                                                     name='label_reverse_lookup_table')
    true_intent_i = label_reverse_lookup.lookup(labels, 'true_intent_i')
    with tf.control_dependencies([tf.assert_equal(true_intent_i < 0, False)]):
        batch_loss = tf.losses.sparse_softmax_cross_entropy(true_intent_i, logits)
    # metrics
    eval_op_dict = {
        'accuracy' : tf.metrics.accuracy(labels=labels, predictions=predicted_intent)
    }
    if mode == tf.estimator.ModeKeys.TRAIN:
        l2_lambda = params['l2_lambda']
        
        if l2_lambda > 0:
            l2_loss = tf.add(tf.nn.l2_loss(out_W, 'out_W_L2'), tf.nn.l2_loss(out_b, 'out_b_L2'), 'l2_loss')
            batch_loss += tf.multiply(l2_lambda, l2_loss, name='regularized_loss')
        for tv in tf.trainable_variables():
            tf.summary.histogram(tv.name, tv)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(batch_loss, global_step=global_step, name='optimizer')
        return tf.estimator.EstimatorSpec(
            mode,
            loss=batch_loss,
            train_op=train_op,
            eval_metric_ops=eval_op_dict,
            scaffold=tf.train.Scaffold(init_fn=partial(_init_fn, params=params))
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=batch_loss,
            eval_metric_ops=eval_op_dict,
        )
    else:
        raise NotImplementedError('Unknown mode %s' % mode)
        

def _init_fn(sc, sess, params):
    with tf.variable_scope('', reuse=True):
        emb_weights = tf.get_variable('word_embeddings')
        pretrained_path = params['pretrained_path']
        with tf.gfile.Open(pretrained_path, mode='rb') as pretrained_inp:
            pretrained_emb_weights = np.load(pretrained_inp)
        tf.logging.info("Loaded pretrained embeddings weights %s %s from %s",
                        pretrained_emb_weights.shape, pretrained_emb_weights.dtype,
                        pretrained_path)
        emb_weights.load(pretrained_emb_weights, sess)