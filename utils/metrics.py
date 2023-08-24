import tensorflow as tf


def r2(y_true, y_pred):
    # Unroll inputs
    y_true = tf.reshape(y_true, shape=[-1])
    y_pred = tf.reshape(y_pred, shape=[-1])

    # Get non-zero y_true values as mask
    zero = tf.constant(0, dtype=tf.float32)
    non_zero_mask = tf.not_equal(y_true, zero)
    y_true = tf.boolean_mask(y_true, non_zero_mask)
    y_pred = tf.boolean_mask(y_pred, non_zero_mask)

    # Calculate r2
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - (unexplained_error/(total_error + tf.keras.backend.epsilon())) # Add small value to avoid division by zero