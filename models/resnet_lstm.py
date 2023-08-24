from tensorflow.keras import Model
from models.resnet_block import ResNetBlock
from models.lstm_block import LstmBlock

class ResNetLstm:

    @staticmethod
    def get_model_with_head(model_params) -> Model:
        # Create ResNet block
        block = ResNetBlock(**model_params)
        _res_block_model = block.get_model()

        # Create LSTM model
        lstm_block = LstmBlock(_res_block_model, **model_params)
        model = lstm_block.get_model_with_head(**model_params)
        model._name = model_params['name']
        return model
