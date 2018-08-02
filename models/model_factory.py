from models.frame_level_cnn import CNNModel, BinaryCNNModel
from models.frame_level_lstm import LSTMModel, BinaryLSTMModel
from models.frame_level_mlp import MLPModel, BinaryMLPModel


class ModelFactory:
    def __init__(self):
        self.model_dict = {'CNNModel': CNNModel(), 'LSTMModel': LSTMModel(), 'MLPModel': MLPModel(),
                           'BinaryCNNModel': BinaryCNNModel(), 'BinaryLSTMModel': BinaryLSTMModel(),
                           'BinaryMLPModel': BinaryMLPModel()}

    def get_model(self, model_name):
        return self.model_dict[model_name]
