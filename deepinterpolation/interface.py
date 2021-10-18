import os, sys
import shutil
from pathlib import Path
import numpy as np
import h5py
import datetime
import gc
import multiprocessing as mp

import utils2p

from deepinterpolation.generic import JsonSaver, ClassLoader
from deepinterpolation.generator_collection import CollectorGenerator

def h5py_to_tif(h5py_in_path, tif_out_path=None):
    stack = []
    with h5py.File(h5py_in_path, "r") as file_handle:
        stack = file_handle.get("data")
        stack = np.squeeze(stack)
    if tif_out_path is not None:
        utils2p.save_img(tif_out_path, stack)
    return stack

def find_model(run_dir):
    return [str(dir) for dir in Path(run_dir).rglob("*model.h5")][0]

def find_out_h5(run_dir):
    files = [str(dir) for dir in Path(os.path.join(run_dir, "inference")).rglob("*.h5")]
    if len(files):
        return files[0]
    print("Warning: inference out file not found.")
    return []

def find_old_models(run_dir):
    models = [str(dir) for dir in Path(run_dir).rglob("*.h5")]
    return list(filter(lambda item: not item.endswith("model.h5"), models))

class DefaultInterpolationParams:
    def __init__(self):
        super().__init__()
        # algorithmic parameters
        self.pre_post_frame = 30

        # data selection parameters
        self.N_frames_per_trial = 2000
        self.train_start = 0
        self.train_end = -1
        self.test_start = 0
        self.test_end = 100
        self.pre_post_omission = 0

        # training parameters
        self.generator_name = "SingleTifGeneratorRandomX"
        self.nb_times_through_data = 1
        self.learning_rate = 0.0001
        self.apply_learning_decay = 0
        self.batch_size = 4
        self.steps_per_epoch = 5
        self.nb_gpus = 1
        self.nb_workers = 16
        self.limited_ram = 320*320

        # inference parameters
        self.inference_generator_name = "SingleTifGenerator"

def prepare_data(train_data_tifs, out_data_tifs, offset=(None,None), size=(320, 640)):
    if not isinstance(train_data_tifs, list):
        train_data_tifs = [train_data_tifs]
    if not isinstance(out_data_tifs, list):
        out_data_tifs = [out_data_tifs]
    assert len(train_data_tifs) == len(out_data_tifs)

    stacks = [utils2p.load_img(path) for path in train_data_tifs]
    N_frames, N_y, N_x = stacks[0].shape
    image_size = [N_y, N_x]

    if size is None:
        size = (N_y, N_x)

    if not isinstance(offset, list):
        offset = [offset]

    if len(offset) == 1:
        tmp = [0, 0]
        for i, off in enumerate(offset[0]):
            if off is None:
                tmp[i] = int((image_size[i] - size[i]) / 2)

        offset = [tmp for _ in train_data_tifs]
    assert len(offset) == len(train_data_tifs)
    assert all(np.array(size).flatten() % 32 == 0)
    
    stacks = [stack[:, off[0]:off[0]+size[0], off[1]:off[1]+size[1]] 
                for (stack, off) in zip(stacks, offset)]

    for out_path, stack in zip(out_data_tifs, stacks):
        path, _ = os.path.split(out_path)
        if not os.path.isdir(path):
            os.makedirs(path)
        utils2p.save_img(out_path, stack)

def train(train_data_tifs, run_base_dir, run_identifier, test_data_tifs=None, params=DefaultInterpolationParams(),
          return_dict_run_dir=None):
    if not isinstance(train_data_tifs, list):
        train_data_tifs = [train_data_tifs]
    N_train_files = len(train_data_tifs)
    if test_data_tifs is None:
        test_data_tifs = train_data_tifs
    elif not isinstance(test_data_tifs, list):
        test_data_tifs = [test_data_tifs]

    if not os.path.isdir(run_base_dir):
            os.makedirs(run_base_dir)

    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}

    # define training generator parameters
    generator_param["type"] = "generator"
    generator_param["steps_per_epoch"] = params.steps_per_epoch
    generator_param["name"] = params.generator_name
    generator_param["pre_post_frame"] = params.pre_post_frame
    generator_param["batch_size"] = params.batch_size
    generator_param["start_frame"] = params.train_start
    generator_param["end_frame"] = params.train_end
    generator_param["N_train"] = params.N_frames_per_trial
    generator_param["pre_post_omission"] = params.pre_post_omission
    generator_param["limited_ram"] = params.limited_ram

    # define testing generator parameters
    generator_test_param["type"] = "generator"
    generator_test_param["name"] = params.generator_name
    generator_test_param["pre_post_frame"] = params.pre_post_frame
    generator_test_param["batch_size"] = params.batch_size
    generator_test_param["start_frame"] = params.test_start
    generator_test_param["end_frame"] = params.test_end
    generator_test_param["pre_post_omission"] = params.pre_post_omission
    generator_test_param["steps_per_epoch"] = params.steps_per_epoch
    generator_test_param["limited_ram"] = params.limited_ram

    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param["name"] = "unet_single_1024" 

    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = run_uid
    training_param["batch_size"] = params.batch_size
    training_param["steps_per_epoch"] = params.steps_per_epoch
    training_param["period_save"] = 25
    training_param["nb_gpus"] = params.nb_gpus
    training_param["apply_learning_decay"] = params.apply_learning_decay
    training_param["nb_times_through_data"] = params.nb_times_through_data
    training_param["learning_rate"] = params.learning_rate
    training_param["pre_post_frame"] = params.pre_post_frame
    training_param["loss"] = "mean_absolute_error"
    training_param["nb_workers"] = params.nb_workers
    training_param["model_string"] = (run_identifier + network_param["name"] + "_"
                                      + training_param["loss"] + "_" + training_param["run_uid"])
    training_param["output_dir"] = os.path.join(run_base_dir, training_param["model_string"])

    run_dir = training_param["output_dir"]
    try:
        os.mkdir(run_dir)
    except:
        print("folder already exists")

    # save all parameters as json files, to be able to reproduce
    path_training = os.path.join(run_dir, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_network = os.path.join(run_dir, "network.json")
    json_obj = JsonSaver(network_param)
    json_obj.save_json(path_network)

    path_generators = []
    path_test_generators = []
    for i_ds, train_path in enumerate(train_data_tifs):
        path_generators.append(os.path.join(run_dir, "generator_{}.json".format(i_ds)))
        generator_param["train_path"] = train_path
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generators[i_ds])
    for i_ds, train_path in enumerate(test_data_tifs):
        path_test_generators.append(os.path.join(run_dir, "test_generator_{}.json".format(i_ds)))
        generator_test_param["train_path"] = train_path
        json_obj = JsonSaver(generator_test_param)
        json_obj.save_json(path_test_generators[i_ds])


    # load params from json
    # We find the generator obj in the collection using the json file
    generator_objs = [ClassLoader(this_path) for this_path in path_generators]
    generator_test_objs = [ClassLoader(this_path) for this_path in path_test_generators]

    # We find the network obj in the collection using the json file
    network_obj = ClassLoader(path_network)

    # We find the training obj in the collection using the json file
    trainer_obj = ClassLoader(path_training)

    # We build the generators object. This will, among other things, calculate normalizing parameters.
    train_generators = [generator_obj.find_and_build()(this_path) 
                        for generator_obj, this_path in zip(generator_objs, path_generators)]
    test_generators = [generator_test_obj.find_and_build()(this_path) 
                        for generator_test_obj, this_path in zip(generator_test_objs, path_test_generators)]

    train_generator = CollectorGenerator(train_generators)
    test_generator = CollectorGenerator(test_generators)


    # We build the network object. This will, among other things, calculate normalizing parameters.
    network_callback = network_obj.find_and_build()(path_network)

    # We build the training object.
    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, network_callback, path_training
    )

    # Start training. This can take very long time.
    print("START TRAINING")
    training_class.run()

    print("FINISHED TRAINING")

    # Finalize and save output of the training.
    training_class.finalize()

    if return_dict_run_dir is not None:
        try:
            return_dict_run_dir[0] = run_dir
            return return_dict_run_dir
        except:
            pass
    return run_dir


def inference(data_tifs, run_dir, tif_out_dirs, params=DefaultInterpolationParams()):
    if not isinstance(data_tifs, list):
        data_tifs = [data_tifs]
    if not isinstance(tif_out_dirs, list):
        tif_out_dirs = [tif_out_dirs]
    assert len(data_tifs) == len(tif_out_dirs)

    model_dir = find_model(run_dir)
    inference_dir = os.path.join(run_dir, "inference")

    generator_param = {}
    inferrence_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = params.inference_generator_name
    generator_param["pre_post_frame"] = params.pre_post_frame
    generator_param["pre_post_omission"] = params.pre_post_omission
    generator_param["steps_per_epoch"] = params.steps_per_epoch
    generator_param["batch_size"] = params.batch_size
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1
    generator_param["randomize"] = 0

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"
    inferrence_param["model_path"] = model_dir

    try:
        os.mkdir(inference_dir)
    except:
        print("folder already exists")

    for i_file, (data_tif, tif_out_dir) in enumerate(zip(data_tifs, tif_out_dirs)):
        print("Denoising: " + data_tif)
        generator_param["train_path"] = data_tif

        _, file_name = os.path.split(data_tif)
        inferrence_param["output_file"] = os.path.join(inference_dir, "denoised_"+file_name[:-4]+".h5")

        path_generator = os.path.join(inference_dir, "generator.json")
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)

        path_infer = os.path.join(inference_dir, "inferrence.json")
        json_obj = JsonSaver(inferrence_param)
        json_obj.save_json(path_infer)

        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)

        inferrence_obj = ClassLoader(path_infer)
        inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)

        inferrence_class.run()

        _ = h5py_to_tif(inferrence_param["output_file"], tif_out_dir)
        print("Saved to: " + tif_out_dir)

def clean_up(run_dir, tmp_data_dirs):
    if not isinstance(tmp_data_dirs, list):
        tmp_data_dirs = [tmp_data_dirs]
    _ = [os.remove(tmp_data_dir) for tmp_data_dir in tmp_data_dirs]

    old_models = find_old_models(run_dir)
    _ = [os.remove(old_model) for old_model in old_models]

    tmp_out_h5 = find_out_h5(run_dir)
    if isinstance(tmp_out_h5, list):
        _ = [os.remove(_) for _ in tmp_out_h5]
    else:
        os.remove(tmp_out_h5)

def copy_run_dir(run_dir, target_dir, delete_tmp=False):
    # if not os.path.isdir(target_dir):
    #     os.makedirs(target_dir)
    # TODO: Include fix that allows copying if directory exists in <3.8
    try:
        version = sys.version_info
        if version[0] >= 3 and version[1] >= 8:  # check python version number
            shutil.copytree(src=run_dir, dst=target_dir, dirs_exist_ok=True)
        else:
            shutil.copytree(src=run_dir, dst=target_dir)
        if delete_tmp:
            shutil.rmtree(run_dir)
    except FileExistsError:
        print("Copying from \n"+run_dir+" to \n"+target_dir+\
              "\ndid not work because the target directory already exists." +\
              "\nPlease try again manually.")
    except TypeError:
        print("Copying from \n"+run_dir+" to \n"+target_dir+\
              "\ndid not work because of an error with the python version (this code works with python >=3.8)." +\
              "\nPlease try again manually.")


    
if __name__ == "__main__":
    ONEDS_COMPLETE = False
    COMPARE = False
    TEST_CLEANUP = True

    if ONEDS_COMPLETE:
        data_dir = "/mnt/NAS/JB/210301_J1xCI9/Fly1/012_xz/processed/green_com_warped.tif"
        run_data_dir = "/home/jbraun/bin/deepinterpolation/tmpdata/210301_J1xCI9_Fly1_001_xz.tif"
        data_out_dir = "/mnt/NAS/JB/210301_J1xCI9/Fly1/001_xz/processed/green_denoised.tif"
        run_base_dir = "/home/jbraun/bin/deepinterpolation/runs"
        run_identifier = "test_210301"

        prepare_data(data_dir, run_data_dir)
        run_dir = train(run_data_dir, run_base_dir, run_identifier)
        inference(run_data_dir, run_dir, data_out_dir)

    if COMPARE:
        data_base_dir = "/home/jbraun/bin/deepinterpolation/sample_data"
        run_base_dir = "/home/jbraun/bin/deepinterpolation/runs"
        data_tifs = [
            os.path.join(data_base_dir, "longterm_003_crop.tif"),
            os.path.join(data_base_dir, "R57C10_GCaMP6s_8Hz_crop.tif"),
            os.path.join(data_base_dir, "210301_001_crop_8Hz.tif"),
            os.path.join(data_base_dir, "ABO_GCaMP6s_16Hz_crop.tif")
        ]
        out_dirs = [
            os.path.join(data_base_dir, "comp_denoise_longterm_003_crop.tif"),
            os.path.join(data_base_dir, "comp_denoise_R57C10_GCaMP6s_8Hz_crop.tif"),
            os.path.join(data_base_dir, "comp_denoise_210301_001_crop_8Hz.tif"),
            os.path.join(data_base_dir, "comp_denoise_ABO_GCaMP6s_16Hz_crop.tif")
        ]
        run_identifiers = [
            "comp_denoise_longterm",
            "comp_denoise_R57C10_GCaMP6s_8Hz",
            "comp_denoise_210301_001_8Hz",
            "comp_denoise_ABO_GCaMP6s_16Hz.tif"
        ]

        for i_run, (data_tif, out_dir, run_identifier) in enumerate(zip(data_tifs, out_dirs, run_identifiers)):
            run_dir = train(data_tif, run_base_dir=run_base_dir, run_identifier=run_identifier)
            inference(data_tif, run_dir=run_dir, tif_out_dirs=out_dir)
            gc.collect()

    if TEST_CLEANUP:
        fly_dir = "/mnt/NAS/JB/210301_J1xCI9/Fly1"
        trial_dirs = [os.path.join(fly_dir, "{:03d}_xz".format(i+1)) for i in range(12)]
        tmp_dir = "/home/jbraun/tmp/deepinterpolation/runs"
        tmp_dirs = [os.path.join(tmp_dir, folder) for folder in os.listdir(tmp_dir)]
        i_trials = [int(tmp_d[60:62])-1 for tmp_d in tmp_dirs]
        i_trials_sort = np.argsort(i_trials)
        for i, trial_dir in enumerate(trial_dirs):
            tmp_run_dir = tmp_dirs[i_trials_sort[i]]
            copy_run_dir(tmp_run_dir, os.path.join(trial_dir, "processed", "denoising_run"),
                         delete_tmp=False)

