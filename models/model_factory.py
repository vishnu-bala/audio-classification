from models.frame_level_cnn import AudioCNNModel
from models.frame_level_lstm import LSTMModel
from models.frame_level_mlp import MLPModel


class ModelFactory:
    def __init__(self):
        self.model_dict = {'AudioCNNModel': AudioCNNModel(), 'LSTMModel': LSTMModel(), 'MLPModel': MLPModel()}

    def get_model(self, model_name):
        return self.model_dict[model_name]
