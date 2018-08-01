import tensorflow as tf
from tensorflow import gfile, flags, logging
from tensorflow.python.platform.flags import FLAGS

import utils


def resize_axis(tensor, axis, new_size, fill_value=0):
    """
    Truncates or pads a tensor to new_size on a given axis such that
    tensor.shape[axis] == new_size. If the size increases, the padding
    will be performed at the end, using fill_value.

    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.

    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)

    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


def get_audio_feature_matrix(features,
                             feature_size,
                             max_frames,
                             max_quantized_value,
                             min_quantized_value):
    """
    Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames


def read_and_convert(tfrecord_data_path, feature_names, feature_sizes, num_classes, max_frames=300,
                     max_quantized_value=2, min_quantized_value=-2):
    # grab the tensorflow session
    with tf.Session() as sess:
        list_of_feature_names = [
            feature_names.strip() for feature_names in feature_names.split(',')]
        list_of_feature_sizes = [
            int(feature_sizes) for feature_sizes in feature_sizes.split(',')]

        files = gfile.Glob(tfrecord_data_path)
        if not files:
            raise IOError("Unable to find training files. tfrecord_data_path='" +
                          tfrecord_data_path + "'.")
        logging.info("Number of training files: %s.", str(len(files)))
        filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)

        # training_data = [
        #     reader.prepare_reader(filename_queue) for _ in range(num_readers)
        # ]

        reader = tf.TFRecordReader()
        filename, serialized_example = reader.read(filename_queue)

        contexts, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={"video_id": tf.FixedLenFeature(
                [], tf.string),
                "labels": tf.VarLenFeature(tf.int64)},
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
                for feature_name in list_of_feature_names
            })

        # read ground truth labels
        labels = (tf.cast(
            tf.sparse_to_dense(contexts["labels"].values, (num_classes,), 1,
                               validate_indices=False),
            tf.int32))

        num_features = len(list_of_feature_names)

        # loads different types of features in the feature_lists and concatenates them
        feature_matrices = [None] * num_features
        for feature_index in range(num_features):
            feature_matrix, num_frames_in_this_feature = get_audio_feature_matrix(
                features[list_of_feature_names[feature_index]],
                list_of_feature_sizes[feature_index],
                max_frames,
                max_quantized_value,
                min_quantized_value)
            # add to the feature_matrices list
            feature_matrices[feature_index] = feature_matrix

        # concatenate different features
        audio_matrices = tf.concat(feature_matrices, 1)

        batch_video_ids, batch_audio_matrices, batch_labels = tf.train.shuffle_batch(
            [tf.expand_dims(contexts["video_id"], 0), tf.expand_dims(audio_matrices, 0), tf.expand_dims(labels, 0)],
            batch_size=1, capacity=1 * 3, num_threads=1, min_after_dequeue=1)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # create a Coordinator and run all QueueRunners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        count_tfrecord = 0
        # Run the tensorflow session to read from the tfrecord files..
        try:
            while not coord.should_stop():
                video_id, audio_feature_matrix, label = sess.run([batch_video_ids, batch_audio_matrices, batch_labels])
                count_tfrecord = count_tfrecord + 1
                # print the count of tfrecord
                print('TFRecord count: {}'.format(count_tfrecord))
                # print context
                print('Context:')
                print('video_id: {}'.format(video_id))
                print('label: {}'.format(label))
                # These are identified as labels containing water samples
                # 288, 293, 370, 371, 372, 56
                if label[0][0][56] == 1 or label[0][0][288] == 1 or label[0][0][293] == 1 \
                        or label[0][0][370] == 1 or label[0][0][371] == 1 or label[0][0][372] == 1:
                    print('Water sample found. video id: {}'.format(video_id))
                else:
                    print('Non-water sample found. video id: {}'.format(video_id))
                # print feature lists
                # print('\nFeature Lists:')
                # print('audio_feature_matrix: {}'.format(audio_feature_matrix))
        except tf.errors.OutOfRangeError:
            print("Done reading tfrecords")

        # request to stop the threads
        coord.request_stop()
        # wait for the threads to stop
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    # command line option..
    flags.DEFINE_string(
        "tfrecord_data_path", "",
        "File glob for the tfrecord data files. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "audio_embedding", "Name of the feature "
                                                            "to use for training.")
    flags.DEFINE_string("feature_sizes", "128", "Length of the feature vectors.")
    flags.DEFINE_integer("num_classes", 527, "Number of classes in dataset.")

    read_and_convert(FLAGS.tfrecord_data_path, FLAGS.feature_names, FLAGS.feature_sizes, FLAGS.num_classes)
