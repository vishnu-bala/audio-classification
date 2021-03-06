from keras.layers import LSTM, Dropout, Dense, Activation, Input, BatchNormalization, Flatten
from keras import Model


class MLPModel:
    def __init__(self):
        pass

    def create_model(self, model_input, common_config, model_config):

        inputs = Input(tensor=model_input)

        model_name = next(iter(model_config))
        layers = model_config[model_name]['layers']

        x = None
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


class BinaryMLPModel:
    def __init__(self):
        pass

    def create_model(self, model_input, common_config, model_config):

        inputs = Input(tensor=model_input)

        model_name = next(iter(model_config))
        layers = model_config[model_name]['layers']

        x = None
        for layer_index, layer in enumerate(layers):
            class_name = next(iter(layer))

            kwargs = layer[class_name]

            if layer_index == 0:
                x = globals()[class_name](**kwargs)(inputs)
            else:
                x = globals()[class_name](**kwargs)(x)

        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
