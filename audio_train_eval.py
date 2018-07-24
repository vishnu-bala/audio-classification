import os
import tensorflow as tf
from tensorflow import logging
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import flags
from tensorflow.python.platform.flags import FLAGS

from models.model_factory import ModelFactory
from utils import get_input_evaluation_tensors, get_input_data_tensors, get_reader

OUTPUT_MODEL_FILENAME = 'model.json'
OUTPUT_WEIGHTS_FILENAME = 'weights.h5'

LSTM_DEFAULT_CONFIG = {
    "LSTMModel": {
        "layers": [
            {
                "LSTM": {
                    "return_sequences": True,
                    "units": 256
                }
            },
            {
                "Dropout": {
                    "rate": 0.3
                }
            },
            {
                "LSTM": {
                    "return_sequences": True,
                    "units": 512
                }
            },
            {
                "Dropout": {
                    "rate": 0.3
                }
            },
            {
                "LSTM": {
                    "units": 256
                }
            },
            {
                "Dense": {
                    "units": 256
                }
            },
            {
                "Dropout": {
                    "rate": 0.3
                }
            }
        ]
    }
}

AUDIO_CNN_DEFAULT_CONFIG = {
    "AudioCNNModel": {
        "layers": [
            {
                "Conv2D": {
                    "filters": 64,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv1"
                }
            },
            {
                "MaxPooling2D": {
                    "pool_size": 2,
                    "strides": 2,
                    "padding": "same",
                    "name": "pool1"
                }
            },
            {
                "Conv2D": {
                    "filters": 128,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv2"
                }
            },
            {
                "MaxPooling2D": {
                    "pool_size": 2,
                    "strides": 2,
                    "padding": "same",
                    "name": "pool2"
                }
            },
            {
                "Conv2D": {
                    "filters": 256,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv3/conv3_1"
                }
            },
            {
                "Conv2D": {
                    "filters": 256,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv3/conv3_2"
                }
            },
            {
                "MaxPooling2D": {
                    "pool_size": 2,
                    "strides": 2,
                    "padding": "same",
                    "name": "pool3"
                }
            },
            {
                "Conv2D": {
                    "filters": 512,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv3/conv4_1"
                }
            },
            {
                "Conv2D": {
                    "filters": 512,
                    "kernel_size": 3,
                    "strides": 1,
                    "activation": "relu",
                    "padding": "same",
                    "name": "conv3/conv4_2"
                }
            },
            {
                "MaxPooling2D": {
                    "pool_size": 2,
                    "strides": 2,
                    "padding": "same",
                    "name": "pool4"
                }
            },
            {
                "Flatten": {
                    "name": "flatten_"
                }
            },
            {
                "Dense": {
                    "units": 4096,
                    "activation": "relu",
                    "name": "vggish_fc1/fc1_1"
                }
            },
            {
                "Dense": {
                    "units": 4096,
                    "activation": "relu",
                    "name": "vggish_fc1/fc1_2"
                }
            }
        ]
    }
}


class EvaluateInputTensor(Callback):
    """
    Validates a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evaluation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


# def build_prediction_graph(serialized_examples, reader):
#     video_id, model_input_raw, labels_batch, num_frames = (
#         reader.prepare_serialized_examples(serialized_examples))
#     return model_input_raw, labels_batch, num_frames
#

def train_and_eval_from_config(common_config, model_config, model_output_dir):
    # for reproducibility
    tf.set_random_seed(0)

    # Load Data from tfrecord
    # first get the reader based on the config: frame feature reader or
    # aggregated feature reader
    reader = get_reader(common_config['feature_names'],
                        common_config['feature_sizes'],
                        common_config['num_classes'],
                        common_config['frame_features'])
    #
    # if common_config['frame_features']:
    #     serialized_examples = tf.placeholder(tf.string, shape=(None,))
    #
    #     fn = lambda x: build_prediction_graph(x, reader)
    #     model_input_raw, labels_batch, num_frames = (
    #         tf.map_fn(fn, serialized_examples,
    #                   dtype=(tf.float32, tf.bool, tf.int32)))
    # else:
    #     serialized_examples = tf.placeholder(tf.string, shape=(None,))
    #
    #     model_input_raw, labels_batch, num_frames = (
    #         build_prediction_graph(serialized_examples, reader))
    #
    # get the input data tensors
    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            reader,
            common_config['training_data_path'],
            batch_size=common_config['batch_size'],
            num_readers=common_config['num_readers'],
            num_epochs=common_config['num_epochs']
        ))

    tf.summary.histogram("model/input_raw", model_input_raw)

    # TODO
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input_raw = tf.nn.l2_normalize(model_input_raw, feature_dim)

    # reshape model_input_raw and labels_batch to fit as the input to keras model
    # TODO see if we need this line..
    # model_input_raw = tf.reshape(model_input_raw, shape=(batch_size, 1, 300, 128))

    # TODO see if we need this line..
    labels_batch = tf.cast(labels_batch, tf.float32)

    # create model
    model_name = next(iter(model_config))
    train_model = ModelFactory().get_model(model_name).create_model(model_input_raw,
                                                                    common_config,
                                                                    model_config)

    # compile the model
    # Pass the target tensor `labels_batch` to train_model.compile
    # via the `target_tensors` keyword argument
    train_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'],
                        target_tensors=[labels_batch])
    train_model.summary()

    # create a separate evaluation model
    # if common_config['frame_features']:
    #     serialized_examples = tf.placeholder(tf.string, shape=(None,))
    #
    #     fn = lambda x: build_prediction_graph(x, reader)
    #     model_input_raw, labels_batch, num_frames = (
    #         tf.map_fn(fn, serialized_examples,
    #                   dtype=(tf.float32, tf.bool, tf.int32)))
    # else:
    #     serialized_examples = tf.placeholder(tf.string, shape=(None,))
    #
    #     model_input_raw, labels_batch, num_frames = (
    #         build_prediction_graph(serialized_examples, reader))

    video_id_batch, eval_model_input_raw, eval_labels_batch, eval_num_frames = (
        get_input_evaluation_tensors(
            reader,
            common_config['evaluation_data_path'],
            batch_size=common_config['batch_size'],
            num_readers=common_config['num_readers'],
            num_epochs=common_config['num_epochs']))
    tf.summary.histogram("eval/model_input_raw", eval_model_input_raw)

    feature_dim = len(eval_model_input_raw.get_shape()) - 1

    # Normalize input features.
    eval_model_input_raw = tf.nn.l2_normalize(eval_model_input_raw, feature_dim)

    # reshape model_input_raw and labels_batch to fit as the input to keras model

    # TODO see if we need these 2 lines..
    eval_labels_batch = tf.cast(eval_labels_batch, tf.float32)
    # eval_labels_batch = tf.one_hot(eval_labels_batch, num_classes)
    # create model
    eval_model = ModelFactory().get_model(model_name).create_model(eval_model_input_raw, common_config, model_config)

    # compile the eval model
    # Pass the target tensor `eval_labels_batch` to eval_model.compile
    # via the `target_tensors` keyword argument
    eval_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'],
                       target_tensors=[eval_labels_batch])
    eval_model.summary()

    # get the session..
    from keras import backend as K
    sess = K.get_session()

    sess.run(tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # TODO we may have to change the steps_per_epoch here if the training
    # TODO and validation accuracies aren't good..
    train_model.fit(epochs=common_config['num_epochs'],
                    steps_per_epoch=common_config['training_steps_per_epoch'],
                    callbacks=[EvaluateInputTensor(eval_model, steps=common_config['eval_steps_per_epoch'])])

    # save the model
    # 1. serialize model to JSON
    model_json = train_model.to_json()
    model_file_to_save = os.path.join(model_output_dir, OUTPUT_MODEL_FILENAME)
    with open(model_file_to_save, "w") as json_file:
        json_file.write(model_json)

    # 2. save the model weights
    weights_file_to_save = os.path.join(model_output_dir, OUTPUT_WEIGHTS_FILENAME)

    train_model.save_weights(weights_file_to_save)
    logging.info("Saved model and weights to " + model_output_dir)

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()


def train_and_eval(training_data_path, evaluation_data_path, batch_size, num_readers,
                   num_epochs, feature_names, feature_sizes, num_classes, frame_features,
                   model_output_dir, model_name, training_steps_per_epoch, eval_steps_per_epoch):
    cc = {
        "training_data_path": training_data_path,
        "evaluation_data_path": evaluation_data_path,
        "batch_size": batch_size,
        "num_readers": num_readers,
        "num_epochs": num_epochs,
        "feature_names": feature_names,
        "feature_sizes": feature_sizes,
        "num_classes": num_classes,
        "frame_features": frame_features,
        "training_steps_per_epoch": training_steps_per_epoch,
        "eval_steps_per_epoch": eval_steps_per_epoch
    }

    mc = None
    if model_name == 'LSTMModel':
        mc = LSTM_DEFAULT_CONFIG
    else:
        mc = AUDIO_CNN_DEFAULT_CONFIG

    train_and_eval_from_config(cc, mc, model_output_dir)


if __name__ == '__main__':
    # command line common config options..
    flags.DEFINE_string(
        "training_data_path", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string(
        "evaluation_data_path", "",
        "File glob defining the evaluation dataset in tensorflow.SequenceExample "
        "format. The SequenceExamples are expected to have an 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
    flags.DEFINE_integer("num_classes", 4716, "Number of classes in dataset.")
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("training_steps_per_epoch", 10,
                         "The number of iterations per epoch in the training loop.")
    flags.DEFINE_integer("eval_steps_per_epoch", 10,
                         "The number of iterations per epoch in the evaluation loop.")

    # model-specific config
    flags.DEFINE_string(
        "model_name", "LSTMModel",
        "Which architecture to use for the model. Models are defined "
        "in models package")
    flags.DEFINE_string("model_output_dir", "./model_output_dir/",
                        "The directory to save the model files in.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                     "to use for training.")

    # Other unused (for now) flags..
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float("regularization_penalty", 1.0,
                       "How much weight to give to the regularization loss (the label loss has "
                       "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")
    # flags.DEFINE_integer("export_model_steps", 1000,
    #                      "The period, in number of steps, with which the model "
    #                      "is exported for batch prediction.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    # flags.DEFINE_bool(
    #     "log_device_placement", False,
    #     "Whether to write the device on which every op will run into the "
    #     "logs on startup.")

    # Logging the version.
    logging.set_verbosity(tf.logging.DEBUG)
    logging.info("Tensorflow version: %s.", tf.__version__)

    train_and_eval(FLAGS.training_data_path, FLAGS.evaluation_data_path,
                   FLAGS.batch_size, FLAGS.num_readers, FLAGS.num_epochs,
                   FLAGS.feature_names, FLAGS.feature_sizes, FLAGS.num_classes,
                   FLAGS.frame_features, FLAGS.model_output_dir,
                   FLAGS.model_name, FLAGS.training_steps_per_epoch,
                   FLAGS.eval_steps_per_epoch)
