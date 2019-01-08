import tensorflow as tf

def serving_fn():
    # we expect a single message only (no batches) from an upstream model
    receiver_msg_ph = tf.placeholder(tf.string, (None,))
    return tf.estimator.export.ServingInputReceiver(features = {
        # but the model fn works with batches
        'tokens' : tf.expand_dims(receiver_msg_ph, axis=0),
        'tokens_len' : tf.expand_dims(tf.shape(receiver_msg_ph)[0], axis=0)
    }, receiver_tensors = {
        'preprocessed_msg' : receiver_msg_ph
    })