import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras import Model


class CNNModel:
    def __init__(self):
        pass

    def create_model(self, model_input, common_config, model_config):
        # Block 1
        # model input = [ # audio channel, # number of frames, # number of features ]
        # inputs = Input(shape=(1, 300, 128), name='input')
        model_input = tf.reshape(model_input, shape=(common_config['batch_size'], 1, 300, 128))
        inputs = Input(tensor=model_input)

        # Remaining blocks from config
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

        outputs = Dense(common_config['num_classes'], activation='softmax', name='vggish_fc2')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
