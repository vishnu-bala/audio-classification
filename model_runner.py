import json
import glob
import os
import time
from audio_train_eval import train_and_eval_from_config

from os.path import basename
from tensorflow.python.platform import flags
from tensorflow.python.platform.flags import FLAGS


def make_dir_if_not_exists(time_str, cc_path, cc_index, mc_path):
    cc_file_name = basename(cc_path.split('.')[0])
    mc_file_path = basename(mc_path.split('.')[0])
    dir_path = '{}_{}_{}_{}'.format(time_str, cc_file_name, cc_index, mc_file_path)
    print('dir_path: {}'.format(dir_path))
    cwd = os.getcwd()
    full_dir_path = os.path.join(cwd, 'model_output_dir', dir_path)
    print('full_dir_path: {}'.format(full_dir_path))
    if not os.path.exists(full_dir_path):
        os.mkdir(full_dir_path)
    return full_dir_path


if __name__ == '__main__':
    flags.DEFINE_string("common_config_file", "", "the path to the common config file")
    flags.DEFINE_string("model_config_dir", "", "the path to the model config directory")
    print(FLAGS.common_config_file)
    print(FLAGS.model_config_dir)

    model_config_files = glob.glob(FLAGS.model_config_dir)

    # get current time
    current_time_str = time.strftime("%Y%m%d_%H%M")

    with open(FLAGS.common_config_file) as common_config_file:
        common_configs = json.load(common_config_file)
        for cc_index, common_config in enumerate(common_configs):
            print(common_config)
            print('----------------------------------------------------------------------------------')
            for model_config_file in model_config_files:
                with open(model_config_file) as model_config_file_handler:
                    model_config = json.load(model_config_file_handler)
                    print(model_config)
                    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    # create output directory
                    model_output_dir = make_dir_if_not_exists(current_time_str,
                                                              FLAGS.common_config_file, cc_index, model_config_file)

                    train_and_eval_from_config(common_config, model_config, model_output_dir)

                    # train_and_eval(common_config['training_data_path'],
                    #                common_config['evaluation_data_path'],
                    #                common_config['batch_size'],
                    #                common_config['num_readers'],
                    #                common_config['num_epochs'],
                    #                common_config['feature_names'],
                    #                common_config['feature_sizes'],
                    #                common_config['num_classes'],
                    #                common_config['frame_features'],
                    #                model_output_dir,
                    #                next(iter(model_config)),
                    #                common_config['training_steps_per_epoch'],
                    #                common_config['eval_steps_per_epoch'])
