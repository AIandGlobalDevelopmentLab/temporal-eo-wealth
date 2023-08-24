import tensorflow as tf
from tensorflow.keras import Model
from models.resnet_block import ResNetBlock

class ResNet:

    @staticmethod
    def get_model_with_head(model_params) -> Model:
        # Create ResNet block
        block = ResNetBlock(**model_params)
        _res_block_model = block.get_model_with_head(**model_params)

        _res_block_model._name = model_params['name']
        return _res_block_model

    @staticmethod
    def get_model_with_classification_head(model_params) -> Model:
        # Create ResNet block
        block = ResNetBlock(**model_params)
        _res_block_model = block.get_model_with_classification_head(**model_params)

        _res_block_model._name = model_params['name']
        return _res_block_model
