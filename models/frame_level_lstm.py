from keras.layers import LSTM, Dropout, Dense, Activation, Input
from keras import Model

class LSTMModel:
    def __init__(self):
        pass

    def create_model(self, model_input, common_config, model_config):
        # x = LSTM(256, input_shape=(model_input.shape[0],
        #                            model_input.shape[1],
        #                            model_input.shape[2],
        #                            model_input.shape[3]),
        #          return_sequences=True)(model_input)
        inputs = Input(tensor=model_input)

        model_name = next(iter(model_config))
        layers = model_config[model_name]['layers']

        x = None
        # x = LSTM(256, return_sequences=True)(inputs)
        # x = Dropout(0.3)(x)
        # x = LSTM(512, return_sequences=True)(x)
        # x = Dropout(0.3)(x)
        # x = LSTM(256)(x)
        # x = Dense(256)(x)
        # x = Dropout(0.3)(x)

        for layer_index, layer in enumerate(layers):
            class_name = next(iter(layer))

            kwargs = layer[class_name]

            if layer_index == 0:
                x = globals()[class_name](**kwargs)(inputs)
            else:
                x = globals()[class_name](**kwargs)(x)

        x = Dense(common_config['num_classes'])(x)
        outputs = Activation('softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
