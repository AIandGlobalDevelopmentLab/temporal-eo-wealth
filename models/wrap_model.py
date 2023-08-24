import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, TimeDistributed, Input
from tensorflow.keras.models import Model

N_FRAMES_IN_TS_IMG = 10  # A time-series image has ten frames, each corresponding to a three year period

class MakeWindowsLayer(Layer):

    def __init__(self, window_size, name='window_split_layer'):
        super(MakeWindowsLayer, self).__init__(name=name)
        self.window_size = window_size

    def build(self, input_shape):
        self.start_ixs = tf.range(N_FRAMES_IN_TS_IMG - self.window_size + 1)

    def call(self, inputs):
        get_window = lambda start: inputs[:, slice(start, start + self.window_size, None)]
        windows = tf.map_fn(fn=get_window, elems=self.start_ixs, fn_output_signature=tf.float32)
        windows = tf.ensure_shape(windows, (6, None, 5, 224, 224, 8))
        windows = tf.transpose(windows, perm=(1, 0, 2, 3, 4, 5))
        return windows


class MergeWindowsLayer(Layer):

    def __init__(self, name='merge_layer'):
        super(MergeWindowsLayer, self).__init__(name=name)

    def build(self, input_shape):
        self.window_size = input_shape[2]
        self.start_ixs = tf.range(N_FRAMES_IN_TS_IMG - self.window_size + 1)
        
        if self.window_size > (N_FRAMES_IN_TS_IMG // 2):
            width = N_FRAMES_IN_TS_IMG - self.window_size + 1
        else:
            width = self.window_size
        start_c = tf.range(1, width + 1)
        end_c = tf.reverse(start_c, axis=[0])
        middle_c = tf.constant(width, shape=(N_FRAMES_IN_TS_IMG - 2 * width))
        weights = tf.concat([start_c, middle_c, end_c], axis=0)
        self.pred_weights = tf.cast(weights, tf.float32)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        pad_fn = lambda i: tf.pad(inputs[:, i], ((0,0), (i, N_FRAMES_IN_TS_IMG - self.window_size - i), (0,0)))
        padded_inputs = tf.map_fn(fn=pad_fn, elems=self.start_ixs, fn_output_signature=tf.float32)
        padded_inputs = tf.transpose(padded_inputs, perm=(1, 0, 2, 3))
        return tf.reduce_sum(padded_inputs, axis=[1, 3]) / self.pred_weights

def _get_wrapped_resnet(base_model):
    model_input = tf.keras.Input(shape=(10, 224, 224, 8), name='model_input')
    outputs = tf.keras.layers.TimeDistributed(base_model)(model_input)
    return Model(inputs=model_input, outputs=outputs, name='wrapped_resnet_model')
    
def get_wrapped_model(path):
    base_model = load_model(path)
    if base_model.name == 'resnet':
        return _get_wrapped_resnet(base_model)
    window_size = base_model.input_shape[1]
    if window_size == 10:
        return base_model
    input_shape = tuple([N_FRAMES_IN_TS_IMG]) + base_model.input_shape[2:]
    inputs = Input(shape=input_shape, name='model_input')
    x = MakeWindowsLayer(window_size)(inputs)
    x = TimeDistributed(base_model, name='{}_frame_model'.format(window_size))(x)
    x = MergeWindowsLayer()(x)
    return Model(inputs=inputs, outputs=x, name='wrapped_{}_frame_model'.format(window_size))