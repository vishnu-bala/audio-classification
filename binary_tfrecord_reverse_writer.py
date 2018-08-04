import errno
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import gfile, flags, logging
from tensorflow.python.platform.flags import FLAGS

from multiclass_tfrecord_reader import read_and_return_stats


def get_features_and_labels(feature_names, input_tfrecord_data_path, num_classes):
    """
    Utility function to get the features and labels from the multiclass
    samples' tfrecords

    :param feature_names:
    :param input_tfrecord_data_path:
    :param num_classes:
    :return:
    """
    list_of_feature_names = [
        feature_names.strip() for feature_names in feature_names.split(',')]
    # now read the input tfrecord files from the given path
    files = gfile.Glob(input_tfrecord_data_path)
    if not files:
        raise IOError("Unable to find training files. tfrecord_data_path='" +
                      input_tfrecord_data_path + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    files.reverse()
    filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)
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
    context_video_id = contexts["video_id"]
    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (num_classes,), 1,
                           validate_indices=False),
        tf.int32))
    return context_video_id, features, labels


def read_and_convert_water_samples(input_tfrecord_data_path, binary_tfrecord_data_dir,
                                   feature_names, num_classes, water_samples,
                                   start_file_index, end_file_index):
    """
    The 2 functions (water_samples and non_water_samples) are needed because TFRecordWriter
    does not allow appending to an existing *.tfrecord file.

    :param input_tfrecord_data_path:
    :param binary_tfrecord_data_dir:
    :param feature_names:
    :param num_classes:
    :param water_samples:
    :return:
    """
    global writer
    # grab the tensorflow session
    with tf.Session() as sess:
        context_video_id, features, labels = get_features_and_labels(feature_names,
                                                                     input_tfrecord_data_path,
                                                                     num_classes)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # create a Coordinator and run all QueueRunners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # count_tfrecord = 0
        count_water_samples = 0
        # indices representing water related classes..
        # These are identified as labels containing water samples
        # 288 - 297, 370 - 372, 374, 444 - 446, 448 - 456
        indices_of_water_classes = {288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 370, 371, 372, 374, 444,
                                    445, 446, 448, 449, 450, 451, 452, 453, 454, 455, 456}
        # Run the tensorflow session to read from the tfrecord files..
        try:
            while not coord.should_stop() and count_water_samples < water_samples \
                    and start_file_index >= end_file_index:
                # video_id, audio_features, label = sess.run([batch_video_ids, batch_audio_matrices, batch_labels])
                video_id, audio_features, label = sess.run([context_video_id, features, labels])
                # count_tfrecord = count_tfrecord + 1
                # print the count of tfrecord
                # print('TFRecord count: {}'.format(count_tfrecord))
                indices_of_classes_present = np.where(label == 1)[0]
                if any(x in indices_of_water_classes for x in indices_of_classes_present):
                    # print('Water sample found. video id: {}'.format(video_id))
                    # we will store 256 tfrecords in a file
                    if count_water_samples != 0 and (count_water_samples % 256) == 0:
                        writer.close()
                        sys.stdout.flush()
                    # every 256th record, create a new *.tfrecord file
                    if (count_water_samples % 256) == 0:
                        water_file_suffix = count_water_samples / 256
                        water_filename = 'water_{}.tfrecord'.format(start_file_index)
                        water_file_to_save = os.path.join(binary_tfrecord_data_dir, water_filename)

                        if not os.path.exists(water_file_to_save):
                            writer = tf.python_io.TFRecordWriter(water_file_to_save)
                            print('Created new water sample tfrecord file: {}'.format(water_file_to_save))
                        else:
                            # handle the case where the program is killed and restarted and the file is half written
                            # we don't want to overwrite in that case.. and TFRecordWriter doesn't have append
                            # capability either..
                            water_filename = 'water_{}_{}.tfrecord'.format(start_file_index, 'x')
                            water_file_to_save = os.path.join(binary_tfrecord_data_dir, water_filename)
                            writer = tf.python_io.TFRecordWriter(water_file_to_save)
                            print('Created new water sample tfrecord file: {}'.format(water_file_to_save))

                        start_file_index = start_file_index - 1

                    # label as 1 for water samples
                    water_label = 1
                    # create the context with video_id and label
                    context = tf.train.Features(feature={
                        'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[water_label]))
                    })
                    audio_features_uint8 = tf.decode_raw((audio_features["audio_embedding"]), tf.uint8)
                    audio_feature_lists = tf.train.FeatureLists(feature_list={
                        'audio_embedding': tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[audio_features_uint8.eval().tobytes()]
                                    )
                                )
                            ]
                        )
                    })
                    example = tf.train.SequenceExample(context=context, feature_lists=audio_feature_lists)
                    writer.write(example.SerializeToString())
                    # increment the counter and close the writer if 256 records are written
                    count_water_samples = count_water_samples + 1

        except tf.errors.OutOfRangeError:
            print('Done reading tfrecords')

        finally:
            if writer is not None:
                writer.close()
                sys.stdout.flush()

        print('Total count of water samples: {}'.format(count_water_samples))
        # request to stop the threads
        coord.request_stop()
        # wait for the threads to stop
        coord.join(threads)
        sess.close()


def read_and_convert_non_water_samples(input_tfrecord_data_path, binary_tfrecord_data_dir,
                                       feature_names, num_classes, non_water_samples,
                                       start_file_index, end_file_index):
    """
    The 2 functions (water_samples and non_water_samples) are needed because TFRecordWriter
    does not allow appending to an existing *.tfrecord file.

    :param input_tfrecord_data_path:
    :param binary_tfrecord_data_dir:
    :param feature_names:
    :param num_classes:
    :param non_water_samples:
    :return:
    """
    global writer
    # grab the tensorflow session
    with tf.Session() as sess:
        context_video_id, features, labels = get_features_and_labels(feature_names,
                                                                     input_tfrecord_data_path,
                                                                     num_classes)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # create a Coordinator and run all QueueRunners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # count_tfrecord = 0
        count_non_water_samples = 0
        # indices representing water related classes..
        # These are identified as labels containing water samples
        # 288 - 297, 370 - 372, 374, 444 - 446, 448 - 456
        indices_of_water_classes = {288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 370, 371, 372, 374, 444,
                                    445, 446, 448, 449, 450, 451, 452, 453, 454, 455, 456}
        # Run the tensorflow session to read from the tfrecord files..
        try:
            while not coord.should_stop() and count_non_water_samples < non_water_samples \
                    and start_file_index >= end_file_index:
                # video_id, audio_features, label = sess.run([batch_video_ids, batch_audio_matrices, batch_labels])
                video_id, audio_features, label = sess.run([context_video_id, features, labels])
                # count_tfrecord = count_tfrecord + 1
                # print the count of tfrecord
                # print('TFRecord count: {}'.format(count_tfrecord))
                indices_of_classes_present = np.where(label == 1)[0]
                if not any(x in indices_of_water_classes for x in indices_of_classes_present):
                    # print('Non-water sample found. video id: {}'.format(video_id))
                    # we store 256 tfrecords in a file
                    if count_non_water_samples != 0 and (count_non_water_samples % 256) == 0:
                        writer.close()
                        sys.stdout.flush()
                    # every 256th record, create a new *.tfrecord file
                    if (count_non_water_samples % 256) == 0:
                        non_water_file_suffix = count_non_water_samples / 256
                        non_water_filename = 'non_water_{}.tfrecord'.format(start_file_index)
                        non_water_file_to_save = os.path.join(binary_tfrecord_data_dir, non_water_filename)

                        if not os.path.exists(non_water_file_to_save):
                            writer = tf.python_io.TFRecordWriter(non_water_file_to_save)
                            print('Created new non-water sample tfrecord file: {}'.format(non_water_file_to_save))
                        else:
                            # handle the case where the program is killed and restarted and the file is half written
                            # we don't want to overwrite in that case.. and TFRecordWriter doesn't have append
                            # capability either..
                            non_water_filename = 'non_water_{}_{}.tfrecord'.format(start_file_index, 'x')
                            non_water_file_to_save = os.path.join(binary_tfrecord_data_dir, non_water_filename)
                            writer = tf.python_io.TFRecordWriter(non_water_file_to_save)
                            print('Created new non-water sample tfrecord file: {}'.format(non_water_file_to_save))
                        start_file_index = start_file_index - 1

                    # label as 0 for non-water samples
                    non_water_label = 0
                    # create the context with video_id and label
                    context = tf.train.Features(feature={
                        'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[non_water_label]))
                    })
                    audio_features_uint8 = tf.decode_raw((audio_features["audio_embedding"]), tf.uint8)
                    audio_feature_lists = tf.train.FeatureLists(feature_list={
                        'audio_embedding': tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[audio_features_uint8.eval().tobytes()]
                                    )
                                )
                            ]
                        )
                    })
                    example = tf.train.SequenceExample(context=context, feature_lists=audio_feature_lists)
                    writer.write(example.SerializeToString())
                    count_non_water_samples = count_non_water_samples + 1

        except tf.errors.OutOfRangeError:
            print('Done reading tfrecords')

        finally:
            if writer is not None:
                writer.close()
                sys.stdout.flush()

        print('Total count of non-water samples: {}'.format(count_non_water_samples))
        # request to stop the threads
        coord.request_stop()
        # wait for the threads to stop
        coord.join(threads)
        sess.close()


def main(input_tfrecord_data_path, binary_tfrecord_data_dir,
         feature_names, feature_sizes, num_classes,
         water_samples=-1, non_water_samples=-1,
         create_water_tfrecords=False, create_non_water_tfrecords=False,
         start_file_index=145, end_file_index=106):
    # create output dir if it does not exist
    if not os.path.exists(binary_tfrecord_data_dir):
        try:
            os.makedirs(binary_tfrecord_data_dir)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

    if water_samples == -1 and non_water_samples == -1:
        total, water_samples, non_water_samples = read_and_return_stats(input_tfrecord_data_path,
                                                                        feature_names,
                                                                        feature_sizes,
                                                                        num_classes)
        required_samples = min(water_samples, non_water_samples)
    elif water_samples != -1 and non_water_samples == -1:
        required_samples = water_samples
    elif water_samples == -1 and non_water_samples != -1:
        required_samples = non_water_samples
    else:
        required_samples = min(water_samples, non_water_samples)

    # create water samples' tfrecords
    if create_water_tfrecords:
        read_and_convert_water_samples(input_tfrecord_data_path, binary_tfrecord_data_dir,
                                       feature_names, num_classes, required_samples,
                                       start_file_index, end_file_index)
    # create non-water samples' tfrecords
    if create_non_water_tfrecords:
        read_and_convert_non_water_samples(input_tfrecord_data_path, binary_tfrecord_data_dir,
                                           feature_names, num_classes, required_samples,
                                           start_file_index, end_file_index)


if __name__ == '__main__':
    # command line option..
    flags.DEFINE_string(
        "input_tfrecord_data_path", "",
        "File glob for the tfrecord data files.")
    flags.DEFINE_string(
        "binary_tfrecord_data_dir", "",
        "Output directory to store binary labeled data in tfrecord record format. "
        "Each file will contain 256 tfrecords.")
    flags.DEFINE_string("feature_names", "audio_embedding", "Name of the feature "
                                                            "to use for training.")
    flags.DEFINE_string("feature_sizes", "128", "Length of the feature vectors.")
    flags.DEFINE_integer("num_classes", 527, "Number of classes in dataset.")
    flags.DEFINE_integer("water_samples", -1, "Number of water samples in dataset.")
    flags.DEFINE_integer("non_water_samples", -1, "Number of non-water samples in dataset.")
    flags.DEFINE_bool("create_water_tfrecords", False, "Boolean to specify if water sample "
                                                       "tfrecord files need to be created or not.")
    flags.DEFINE_bool("create_non_water_tfrecords", False, "Boolean to specify if non-water sample tfrecord files "
                                                           "need to be created or not")
    flags.DEFINE_integer("start_file_index", 145, "Starting file suffix for the tfrecord file.")
    flags.DEFINE_integer("end_file_index", 106, "Ending file suffix for the tfrecord file.")

    main(FLAGS.input_tfrecord_data_path, FLAGS.binary_tfrecord_data_dir,
         FLAGS.feature_names, FLAGS.feature_sizes, FLAGS.num_classes,
         FLAGS.water_samples, FLAGS.non_water_samples,
         FLAGS.create_water_tfrecords, FLAGS.create_non_water_tfrecords,
         FLAGS.start_file_index, FLAGS.end_file_index)
