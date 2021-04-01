import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib
import h5py
import numpy as np

import utils2p

def h5py_to_tif(h5py_in_path, tif_out_path=None):
    stack = []
    with h5py.File(h5py_in_path, "r") as file_handle:
        stack = file_handle.get("data")
        stack = np.squeeze(stack)
    if tif_out_path is not None:
        utils2p.save_img(tif_out_path, stack)
    return stack

generator_param = {}
inferrence_param = {}

trials = ["003", "005", "007", "010"]
for i_trial, trial in enumerate(trials):
    base_dir = "/home/jbraun/bin/deepinterpolation/runs/multi_longterm_unet_single_1024_mean_absolute_error_2021_03_19_16_41_2021_03_19_16_41"
    inference_dir = base_dir + "/inference"
    model_dir = base_dir + "/2021_03_19_16_41_multi_longterm_unet_single_1024_mean_absolute_error_2021_03_19_16_41_model.h5"
    h5_out_dir = inference_dir + "/longterm_" + trial + "_crop_out.h5"

    # We are reusing the data generator for training here. Some parameters like steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param["steps_per_epoch"] = 5

    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "longterm_" + trial + "_crop.tif",
    )

    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0  # This is important to keep the order and avoid the randomization used during training


    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inferrence_param[
        "model_path"
    ] = model_dir
    # Replace this path to where you want to store your output file
    inferrence_param[
        "output_file"
    ] = h5_out_dir

    jobdir = inference_dir
    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
    inferrence_class.run()


    _ = h5py_to_tif(inferrence_param["output_file"], "/home/jbraun/bin/deepinterpolation/sample_data/denoised_multi_longterm_random1_01_"+trial+"_crop_out.tif")
