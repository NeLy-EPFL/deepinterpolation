import os
from deepinterpolation.generic import JsonSaver, ClassLoader
# from examples.h5py_to_tif import double_h5py_to_tif
import pathlib
from copy import deepcopy

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

def merge_left_right(left, right, x_size_out):
    assert left.shape == right.shape

    N_frames, size_y, size_x_small = left.shape
    overlap = 2 * size_x_small - x_size_out
    stack_out = np.zeros((N_frames, size_y, x_size_out), dtype=left.dtype)

    interpolate_factor = np.ones_like(left)
    interpolate_factor[:, :, -overlap:] = np.linspace(1,0, overlap)
    stack_out[:, :, :size_x_small] += left * interpolate_factor

    interpolate_factor = np.ones_like(right)
    interpolate_factor[:, :, :overlap] = np.linspace(0,1, overlap)
    stack_out[:, :, -size_x_small:] += right * interpolate_factor

    return stack_out

def double_h5py_to_tif(h5py_left_path, h5py_right_path, x_size_out, tif_out_path=None):
    left = h5py_to_tif(h5py_left_path)
    right = h5py_to_tif(h5py_right_path)

    whole_stack = merge_left_right(left, right, x_size_out)
    if tif_out_path is not None:
        utils2p.save_img(tif_out_path, whole_stack)
    return whole_stack

generator_param = {}
inferrence_param = {}

# We are reusing the data generator for training here. Some parameters like steps_per_epoch are irrelevant but currently needs to be provided
generator_param["type"] = "generator"
generator_param["name"] = "SingleTifGeneratorLeftX"  # "SingleTifGenerator"
generator_param["pre_post_frame"] = 30
generator_param["pre_post_omission"] = 0
generator_param["steps_per_epoch"] = 5

generator_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "sample_data",
    "longterm_003_crop.tif",  # "crop_ophys_tiny_761605196.tif",  # "ophys_tiny_761605196.tif",
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
] = "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/2021_03_12_19_57_longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_model.h5"
# "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/repos/public/deepinterpolation/examples/unet_single_1024_mean_absolute_error_2020_11_12_21_33_2020_11_12_21_33/2020_11_12_21_33_unet_single_1024_mean_absolute_error_2020_11_12_21_33_model.h5"

# Replace this path to where you want to store your output file
inferrence_param[
    "output_file"
] = "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference_halves/longterm_003_crop_out_left.h5"  # "/Users/jeromel/test/ophys_tiny_continuous_deep_interpolation.h5"

jobdir = "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference_halves"  # "/Users/jeromel/test/"

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_generator = os.path.join(jobdir, "generator_left.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence_left.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)

# Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
inferrence_class.run()

print("FINISHED LEFT HALF. WILL NOW DENOISE RIGHT HALF")
# repeat the same thing for the right side of the images
inferrence_param_left = deepcopy(inferrence_param)
generator_param["name"] = "SingleTifGeneratorRightX"

path_generator = os.path.join(jobdir, "generator_right.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

inferrence_param[
    "output_file"
] = "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference_halves/longterm_003_crop_out_right.h5"

path_infer = os.path.join(jobdir, "inferrence_right.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)

# Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
inferrence_class.run()
print("FINISHED RIGHT HALF. WILL NOW MERGE")

# merge the left and right half and make it a tif

_ = double_h5py_to_tif(inferrence_param_left["output_file"], inferrence_param["output_file"], x_size_out=576, 
                       tif_out_path="/home/jbraun/bin/deepinterpolation/sample_data/denoised_halves_longterm_003_crop_out.tif")

