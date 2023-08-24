import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from classification_models.tfkeras import Classifiers


class ResNetBlock:

    IMAGE_SIZE = 224  # Input images are 224 px x 224 px
    DEFAULT_MS_CHANNELS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

    def __init__(self,
                 ms_channels=None,
                 path_to_ms_model=None,
                 path_to_nl_model=None,
                 path_to_resnet_model=None,
                 pretrained_ms=True,
                 pretrained_nl=True,
                 freeze_model=False, 
                 **kwargs):
        
        self.freeze_model = freeze_model
        if path_to_resnet_model:
            transfer_model = tf.keras.models.load_model(path_to_resnet_model)
            if freeze_model:
                for layer in transfer_model.layers:
                    layer.trainable = False
            self._inputs = transfer_model.inputs
            self._resnet_outputs = transfer_model.layers[5].output
        else:
            self.ResNet18, _ = Classifiers.get('resnet18')
            self.img_net_model = self.ResNet18((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), weights='imagenet')
            self.ms_channels = ms_channels if ms_channels else self.DEFAULT_MS_CHANNELS
            self.ms_model = self._get_resnet_model(len(self.ms_channels), keep_rgb=True, path_to_model=path_to_ms_model,
                                                pretrained=pretrained_ms)
            self.ms_model._name = 'ms_model'
            self.nl_model = self._get_resnet_model(1, keep_rgb=False, path_to_model=path_to_nl_model,
                                                pretrained=pretrained_nl)
            #self.path_to_resnet_model = path_to_resnet_model
            self.nl_model._name = 'nl_model'
            #if path_to_resnet_model != None:
            #    self.model = tf.keras.models.load_model(path_to_resnet_model)
            self.model=None
            self._inputs, self._resnet_outputs = self._init_block()

    def _get_resnet_model(self,
                          num_channels,
                          keep_rgb,
                          path_to_model=None,
                          pretrained=True) -> tf.keras.models.Model:
        """
        Creates a tf-keras ResNet-18 model according to some parameters
        :param num_channels: int, the number of input channels to the model
        :param path_to_model: string, path to model file. If a model file is provided, this model will be returned
        :param pretrained: bool, whether to initiate with weights from model pretrained on imagenet. If true, the model
        will use these weights for all layers except the first conv-layer. Since the model is trained on 3 channels
        (RGB) we use their average weights for all channels.
        :param keep_rgb: bool, if true, the rgb channels from the pretrained model will be left as they are (i.e. the
        average will not be used for the first three channels)
        :return: tf.keras.models.Model, A tf-keras ResNet-18 model
        """
        if path_to_model:
            return tf.keras.models.load_model(path_to_model)
        else:
            if pretrained:
                return self._get_pretrained_resnet_model(num_channels, keep_rgb)
            else:
                return self.ResNet18((self.IMAGE_SIZE, self.IMAGE_SIZE, num_channels))

    def _get_pretrained_resnet_model(self, num_channels, keep_rgb):
        model = self.ResNet18((self.IMAGE_SIZE, self.IMAGE_SIZE, num_channels))
        conv_1_weights = self._get_first_conv_weights(num_channels, self.img_net_model, keep_rgb)
        model.layers[3].set_weights([conv_1_weights])
        for i in range(4, len(self.img_net_model.layers)):
            img_net_weights = self.img_net_model.layers[i].get_weights()
            model.layers[i].set_weights(img_net_weights)
        return model

    @staticmethod
    def _get_first_conv_weights(num_channels, img_net_model, keep_rgb) -> tf.Tensor:
        """
        Gets the weights for the first convolutional layer based on pretrained ResNet. Since the model is trained on 3
        channels (RGB) we use their average weights for all channels. The rgb channels from the pretrained model can be
        left as they are (i.e. the average will not be used for the first three channels)
        :param num_channels: int, the number of input channels to the model
        :param img_net_model: tf.keras.models.Model, the pretrained model to base our weights off
        :param keep_rgb: bool, if true, the rgb channels from the pretrained model will be left as they are.
        :return: tf.Tensor, the computed weights for the first convolutional layer
        """
        conv1_weights = img_net_model.layers[3].get_weights()[0].copy()
        rgb_mean = conv1_weights.mean(axis=2, keepdims=True)
        if keep_rgb:
            # Includes RGB channels. Keep those as they are. Set the rest to mean
            num_hs_channels = num_channels - 3
            mean_weights = np.tile(rgb_mean, (1, 1, num_hs_channels, 1))
            conv1_weights *= 3 / num_channels
            mean_weights *= 3 / num_channels
            return tf.concat([conv1_weights, mean_weights], axis=2)
        else:
            mean_weights = np.tile(rgb_mean, (1, 1, num_channels, 1))
            mean_weights /= num_channels
            return tf.convert_to_tensor(mean_weights)

    def _init_block(self) -> (Input, concatenate):
        """
        Sets up the connections for the ResNet block used to build the model. Splits the input into MS and NL channels,
        runs these through the MS and NL ResNets respectively, concatenates the output.
        :return: (tf.keras.layers.Input, tf.keras.layers.concatenate), the input and output layers for the ResNet block
        """
        n_ms_bands = self.ms_model.input_shape[3]
        n_nl_bands = self.nl_model.input_shape[3]
        inputs = Input((self.IMAGE_SIZE, self.IMAGE_SIZE, n_ms_bands + n_nl_bands), name="all_bands_inputs")
        ms_input = Lambda(lambda x: x[:, :, :, :n_ms_bands], name="ms_inputs")(inputs)
        nl_input = Lambda(lambda x: x[:, :, :, n_ms_bands:], name="nl_inputs")(inputs)
        ms_output = self.ms_model(ms_input)
        nl_output = self.nl_model(nl_input)
        resnet_outputs = concatenate([ms_output, nl_output], axis=1, name="combined_resnet_output")
        return inputs, resnet_outputs

    def get_model(self, batch_norm=True) -> tf.keras.models.Model:
        """
        Creates a model which takes in both MS and NL data, runs them through separate ResNets and
        concatenate the feature vectors.
        :return: tf.keras.models.Model, ResNet block model
        """
        if batch_norm:
            model_outputs = BatchNormalization(name='norm_combined_resnet_output')(self._resnet_outputs)
        else:
            model_outputs = self._resnet_outputs
        return Model(inputs=self._inputs, outputs=model_outputs, name='resnet_block')

    def get_model_with_head(self, l2=0.1, **kwargs):
        """
        Creates a model which takes in both MS and NL data, runs them through separate ResNets,
        concatenates the feature vectors, runs a ridge regression on output a regression value.
        :return: tf.keras.models.Model, ResNet block model with head
        """
        if l2 > 0:
            outputs = Dense(1,
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.keras.regularizers.l2(l=l2),
                            name="ridge_regression"
                            )(self._resnet_outputs)
        else:
            outputs = Dense(1,
                            activation=tf.nn.relu,
                            name="ridge_regression"
                            )(self._resnet_outputs)
        #if self.path_to_resnet_model:
        #    return self.model(inputs=self._inputs, outputs=outputs, name='ms_nl_resnet_model')
        return Model(inputs=self._inputs, outputs=outputs, name='ms_nl_resnet_model')

    def get_model_with_classification_head(self, dropout_rate=0.5, hidden_neurons=128, **kwargs):
        """
        Creates a model which takes in both MS and NL data, runs them through separate ResNets and
        concatenates the feature vectors. The classification model adds a fully-connected layer on
        top of the encoder, plus a sigmoid layer with a binary target.
        :return: tf.keras.models.Model, ResNet block model with head
        """
        features = self._resnet_outputs
        # features = Dropout(dropout_rate)(features)
        features = BatchNormalization()(features)
        if hidden_neurons > 0:
            features = Dense(hidden_neurons, activation="relu")(features)
        outputs = Dense(1,
                        activation='sigmoid',
                        name="label",
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        )(features)
        #if self.path_to_resnet_model:
        #    return self.model(inputs=self._inputs, outputs=outputs, name='ms_nl_resnet_model')
        return Model(inputs=self._inputs, outputs=outputs, name='ms_nl_resnet_model')
