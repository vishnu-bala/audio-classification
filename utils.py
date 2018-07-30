import tensorflow as tf
from tensorflow import logging, gfile

import data_readers


def get_list_of_feature_names_and_sizes(feature_names, feature_sizes):
    """
    Extracts the list of feature names and the dimensionality of each feature
       from string of comma separated values.

    Args:
      feature_names: string containing comma separated list of feature names
      feature_sizes: string containing comma separated list of feature sizes

    Returns:
      List of the feature names and list of the dimensionality of each feature.
      Elements in the first/second list are strings/integers.
    """
    list_of_feature_names = [
        feature_names.strip() for feature_names in feature_names.split(',')]
    list_of_feature_sizes = [
        int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
    if len(list_of_feature_names) != len(list_of_feature_sizes):
        logging.error("length of the feature names (=" +
                      str(len(list_of_feature_names)) + ") != length of feature "
                                                        "sizes (=" + str(len(list_of_feature_sizes)) + ")")

    return list_of_feature_names, list_of_feature_sizes


def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
    Dequantizes the feature from the byte format to the float format.

    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
    """
    Creates the evaluation data tensors by reading the eval data in tfrecord files

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_readers: How many I/O threads to use.
      num_epochs: How many passes to make over the evaluation data. Default is 1

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
    with tf.name_scope("eval_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        logging.info("number of evaluation files: " + str(len(files)))
        filename_queue = tf.train.string_input_producer(files, shuffle=False)
        eval_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
            ]
        return tf.train.batch_join(
            eval_data,
            batch_size=batch_size,
            capacity=3 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1024,
                           num_epochs=None,
                           num_readers=1):
    """
    Creates the input data tensors by reading the training data in tfrecord files

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      num_readers: How many I/O threads to use.

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for training.")
    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find training files. data_pattern='" +
                          data_pattern + "'.")
        logging.info("Number of training files: %s.", str(len(files)))
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs, shuffle=True)
        training_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
            ]
        return tf.train.shuffle_batch_join(
            training_data,
            batch_size=batch_size,
            capacity=batch_size * 5,
            min_after_dequeue=batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def get_reader(feature_names, feature_sizes, num_classes, frame_features):
    """
    Gets the frame feature reader or the aggregated feature reader based on
    the config during the invocation.
    """
    # Convert feature_names and feature_sizes to lists of values.
    feature_names, feature_sizes = get_list_of_feature_names_and_sizes(
        feature_names, feature_sizes)

    if frame_features:
        reader = data_readers.YT8MFrameFeatureReader(
            num_classes=num_classes,
            feature_names=feature_names, feature_sizes=feature_sizes)
    else:
        reader = data_readers.YT8MAggregatedFeatureReader(
            num_classes=num_classes,
            feature_names=feature_names, feature_sizes=feature_sizes)

    return reader


def get_input_test_tensors(reader,
                           data_pattern,
                           batch_size=1024,
                           num_readers=1):
    """
    Creates the test data tensors by reading the test data in tfrecord files

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_readers: How many I/O threads to use.
      num_epochs: How many passes to make over the test data. Default is 1

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for test.")
    with tf.name_scope("test_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the test files.")
        logging.info("number of test files: " + str(len(files)))
        filename_queue = tf.train.string_input_producer(files, shuffle=False)
        test_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
            ]
        return tf.train.batch_join(
            test_data,
            batch_size=batch_size,
            capacity=3 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)