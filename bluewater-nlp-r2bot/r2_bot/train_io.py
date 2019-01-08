from collections import namedtuple
import tensorflow as tf
import os

DatasetRecord = namedtuple('DatasetRecord',
                           ['text', 'tokens'])

BatchedDatasetRecord = namedtuple('BatchedDatasetRecord',
                                  DatasetRecord._fields + ('tokens_len',))

def input_fn(data_dir, tfr_fn, params, repeat=None, shuffle_size=None, cache_ds=False):
    if not isinstance(tfr_fn, (tuple, list)):
        tfr_fn = [tfr_fn]
    ds_path = [os.path.join(data_dir, fn) for fn in tfr_fn]
    tfr_ds = tf.data.TFRecordDataset(ds_path)
    batch_size = params['batch_size']
    result_ds = transform_tfr_dataset(tfr_ds, batch_size, params)
    if cache_ds:
        result_ds = result_ds.cache()
    if shuffle_size:
        result_ds = result_ds.shuffle(shuffle_size, seed=832)
    if repeat:
        result_ds = result_ds.repeat(count=repeat)
    return result_ds


def transform_tfr_dataset(tfr_ds, batch_size, params):
    parsed_ds = tfr_ds.map(_parse_seq_example)
    parsed_ds = parsed_ds.map(_extract_seq_lengths)
    return parsed_ds.padded_batch(
        batch_size,
        padded_shapes=(BatchedDatasetRecord(
            text=(),
            tokens=(None,),
            tokens_len=(),
        ), ()),
        padding_values=(BatchedDatasetRecord(
            # should not matter
            text='',
            # this is really important!
            tokens='<pad>',
            # should not matter
            tokens_len=0,
        ), '')
    )

def _parse_seq_example(ser):
    ctx_dict, seq_dict = tf.parse_single_sequence_example(
        ser,
        context_features={
            'text': tf.FixedLenFeature((), tf.string),
            'intent': tf.FixedLenFeature((), tf.string),
        },
        sequence_features={
            'token': tf.FixedLenSequenceFeature((), dtype=tf.string)
        })
    label = ctx_dict['intent']
    return DatasetRecord(text=ctx_dict['text'], tokens=seq_dict['token']), label


def _extract_seq_lengths(record, labels):
    return (
        BatchedDatasetRecord(*(record + (tf.shape(record.tokens)[0],))),
        labels
    )

def get_ds_size(data_dir, tfr_fn):
    tfr_path = os.path.join(data_dir, tfr_fn)
    size_path = tfr_path[:tfr_path.rfind('.')] + '-size.txt'
    with open(size_path) as inp:
        return int(inp.readline())