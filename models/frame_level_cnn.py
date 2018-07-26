import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras import Model


class CNNModel:
    def __init__(self):
        pass

    def create_model(self, model_input, common_config, model_config):
        # Block 1
        # model input = [ batch_size, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
        # inputs = Input(shape=(1, 300, 128), name='input')
        model_input = tf.reshape(model_input, shape=(common_config['batch_size'], 1, 300, 128))
        inputs = Input(tensor=model_input)

        # x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(inputs)
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(x)
        #
        # # Block 2
        # x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)
        #
        # # Block 3
        # x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
        # x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(x)
        #
        # # Block 4
        # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
        # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(x)
        #
        # # flatten the output from Conv2D layers to feed into 1-dimensional MLP (Dense) layer
        # x = Flatten(name='flatten_')(x)
        # # 3 MLP layers below
        # x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        # x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)

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
