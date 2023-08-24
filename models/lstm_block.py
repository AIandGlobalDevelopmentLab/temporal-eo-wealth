import tensorflow as tf
from typing import Any
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Bidirectional, Activation
from tensorflow.keras.models import Model


class LstmBlock:

    def __init__(self, input_block: Model, n_of_frames: int, n_lstm_units: int=18, lstm_dropout: float=0.4, 
                 bidirectional: bool=True, trainable_input_block: bool=True, **kwargs: Any):
        self.input_block = input_block
        self.trainable_input_block = trainable_input_block
        self.n_of_frames = n_of_frames
        self.n_lstm_units = n_lstm_units
        self.lstm_dropout = lstm_dropout
        self._inputs, self._lstm_outputs = self._init_block(bidirectional)

    def _init_block(self, bidirectional) -> (Input, LSTM):
        """
        Sets up the base for the LSTM block. The result is used to uild a model with or without a regression head
        :return: (tf.keras.layers.Input, tf.keras.layers.LSTM), the input and output layers for the LSTM block
        """
        input_shape = (self.n_of_frames,) + self.input_block.input_shape[1:]
        inputs = Input(shape=input_shape, name='model_input')
        input_block_outputs = TimeDistributed(self.input_block, name='input_block')(inputs)
        if bidirectional:
            lstm_outputs = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, dropout=self.lstm_dropout), name='frame_features')(input_block_outputs)
        else:
            lstm_outputs = LSTM(self.n_lstm_units, return_sequences=True, dropout=self.lstm_dropout, name='frame_features')(input_block_outputs)
        return inputs, lstm_outputs

    def get_model(self) -> Model:
        """
        Returns a model which outputs the feature vector outputs from the LSTM. Can be used for pretext tasks
        :return: tf.keras.Model, A LSTM model without a head
        """
        return Model(inputs=self._inputs, outputs=self._lstm_outputs)

    def get_model_with_head(self, l2: float = 0.01, **kwargs: Any) -> Model:
        """
        Returns a model which outputs a series of regression values. Can be used to predict to predict IWI
        :param l2: float, lambda value for ridge regression
        :return: tf.keras.Model, A LSTM model with a ridge regression head
        """
        x = TimeDistributed(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2=l2)), 
                            name='dense_logits')(self._lstm_outputs)
        outputs = TimeDistributed(Activation(tf.nn.relu, dtype='float32'), name='unmasked_outputs')(x)

        outputs_mask = tf.keras.Input(shape=(self.n_of_frames, 1), name='outputs_mask')
        masked_outputs = tf.keras.layers.Multiply(name='masked_outputs')([outputs, outputs_mask])
        return Model(inputs=[self._inputs, outputs_mask], outputs=masked_outputs)
