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
    "210301_002_crop.tif",  # "crop_ophys_tiny_761605196.tif",  # "ophys_tiny_761605196.tif",
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
] = "/home/jbraun/bin/deepinterpolation/runs/210301_001_crop_2ndunet_single_1024_mean_absolute_error_2021_03_30_16_19_2021_03_30_16_19/2021_03_30_16_19_210301_001_crop_2ndunet_single_1024_mean_absolute_error_2021_03_30_16_19_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_41_2021_03_19_15_41/2021_03_19_15_41_longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_41_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_20_2021_03_19_15_20/2021_03_19_15_20_longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_20_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_14_31_2021_03_19_14_31/2021_03_19_14_31_longterm_unet_single_1024_mean_absolute_error_2021_03_19_14_31_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_12_30_2021_03_19_12_30/2021_03_19_12_30_longterm_unet_single_1024_mean_absolute_error_2021_03_19_12_30_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_15_05_2021_03_18_15_05/2021_03_18_15_05_longterm_unet_single_1024_mean_absolute_error_2021_03_18_15_05_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_14_31_2021_03_18_14_31/2021_03_18_14_31_longterm_unet_single_1024_mean_absolute_error_2021_03_18_14_31_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_12_41_2021_03_18_12_41/2021_03_18_12_41_longterm_unet_single_1024_mean_absolutecd_error_2021_03_18_12_41_model.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/2021_03_12_19_57_longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_model.h5"

# Replace this path to where you want to store your output file
inferrence_param[
    "output_file"
] = "/home/jbraun/bin/deepinterpolation/runs/210301_001_crop_2ndunet_single_1024_mean_absolute_error_2021_03_30_16_19_2021_03_30_16_19/inference/210301_001_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_41_2021_03_19_15_41/inference/longterm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_20_2021_03_19_15_20/inference/longterm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_14_31_2021_03_19_14_31/inference/longterm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_12_30_2021_03_19_12_30/inference/longerm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_15_05_2021_03_18_15_05/inference/longterm_003_crop_out.h5"
#"/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_14_31_2021_03_18_14_31/inference/longterm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_12_41_2021_03_18_12_41/inference/longterm_003_crop_out.h5"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference/longterm_003_crop_out.h5"  

jobdir = "/home/jbraun/bin/deepinterpolation/runs/210301_001_crop_2ndunet_single_1024_mean_absolute_error_2021_03_30_16_19_2021_03_30_16_19/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_41_2021_03_19_15_41/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_15_20_2021_03_19_15_20/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_14_31_2021_03_19_14_31/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_19_12_30_2021_03_19_12_30/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_15_05_2021_03_18_15_05/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_14_31_2021_03_18_14_31/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_18_12_41_2021_03_18_12_41/inference"
# "/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference"

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


_ = h5py_to_tif(inferrence_param["output_file"], "/home/jbraun/tmp/210301_J7xCI9_Fly1_002_xz_denoised_green.tif")
# "/home/jbraun/bin/deepinterpolation/sample_data/denoised_random1_01_12th2000_210301_001_crop_out.tif")
# "/home/jbraun/bin/deepinterpolation/sample_data/denoised_randomlrc1_05_longterm_003_crop_out.tif")
# "/home/jbraun/bin/deepinterpolation/sample_data/denoised_random_longterm_003_crop_out.tif")
# "/home/jbraun/bin/deepinterpolation/sample_data/denoised_longterm_003_crop_out.tif")