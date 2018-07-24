import tensorflow as tf
from keras.engine.saving import model_from_json
from tensorflow.python.platform import flags
from tensorflow.python.platform.flags import FLAGS

import utils
from utils import get_input_test_tensors


def test_from_saved_model(test_data_path, batch_size, num_readers,
                          num_epochs, feature_names, feature_sizes, num_classes, frame_features,
                          model_output_dir, saved_model_file, saved_weights_file):
    # get the session..
    from keras import backend as K
    sess = K.get_session()

    # for reproducibility
    tf.set_random_seed(0)
    # Load Data from tfrecord
    # first get the reader based on the config: frame feature reader or
    # aggregated feature reader
    reader = utils.get_reader(feature_names, feature_sizes, num_classes, frame_features)

    # get input test tensors
    video_id_batch, test_model_input_raw, test_labels_batch, test_num_frames = (
        get_input_test_tensors(
            reader,
            test_data_path,
            batch_size=batch_size,
            num_readers=num_readers,
            num_epochs=num_epochs))
    tf.summary.histogram("test/model_input_raw", test_model_input_raw)

    feature_dim = len(test_model_input_raw.get_shape()) - 1

    # Normalize input features.
    test_model_input_raw = tf.nn.l2_normalize(test_model_input_raw, feature_dim)
    # test_model_input_raw = tf.reshape(test_model_input_raw, shape=(batch_size, 1, 300, 128))

    # TODO see if we need these 2 lines..
    test_labels_batch = tf.cast(test_labels_batch, tf.float32)

    # load json and create model
    if model_output_dir.endswith('/'):
        model_file_to_load = model_output_dir + saved_model_file
        weights_file_to_load = model_output_dir + saved_weights_file
    else:
        model_file_to_load = model_output_dir + '/' + saved_model_file
        weights_file_to_load = model_output_dir + '/' + saved_weights_file

    json_file = open(model_file_to_load, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    test_model = model_from_json(loaded_model_json)
    # load weights into new model
    test_model.load_weights(weights_file_to_load)

    # compile the model
    test_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    test_model.summary()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # evaluate the model using test data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    loss, acc = test_model.evaluate(test_model_input_raw,
                                    test_labels_batch,
                                    steps=100,
                                    verbose=1)

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()

    print('\nTest accuracy: {0}'.format(acc))


if __name__ == '__main__':
    # command line config options..
    flags.DEFINE_string("model_output_dir", "./model_output_dir/",
                        "The directory to save the model files in.")
    flags.DEFINE_string("saved_model_file", "saved_model_file.json",
                        "The name of the JSON file to save the model in.")
    flags.DEFINE_string("saved_weights_file", "saved_weights_file.h5",
                        "The name of the h5 file to save the model weights in.")
    flags.DEFINE_string(
        "test_data_path", "",
        "File glob defining the test dataset in tensorflow.SequenceExample "
        "format. The SequenceExamples are expected to have an 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                     "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
    flags.DEFINE_integer("num_classes", 4716, "Number of classes in dataset.")

    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_string(
        "model_name", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")

    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")

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
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
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

    test_from_saved_model(FLAGS.test_data_path, FLAGS.batch_size, FLAGS.num_readers,
                          FLAGS.num_epochs, FLAGS.feature_names, FLAGS.feature_sizes,
                          FLAGS.num_classes, FLAGS.frame_features, FLAGS.model_output_dir,
                          FLAGS.saved_model_file, FLAGS.saved_weights_file)
