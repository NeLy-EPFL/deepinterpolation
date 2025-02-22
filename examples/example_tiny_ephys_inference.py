import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib

generator_param = {}
inferrence_param = {}

# We are reusing the data generator for training here. Some parameters like steps_per_epoch are irrelevant but currently needs to be provided
generator_param["type"] = "generator"
generator_param["name"] = "EphysGenerator"
generator_param["pre_post_frame"] = 30
generator_param["pre_post_omission"] = 1
generator_param["steps_per_epoch"] = 10

generator_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "sample_data",
    "ephys_tiny_continuous.dat2",
)

generator_param["batch_size"] = 100
generator_param["start_frame"] = 100
generator_param["end_frame"] = 200  # -1 to go until the end.
generator_param[
    "randomize"
] = 0  # This is important to keep the order and avoid the randomization used during training


inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "core_inferrence"

# Replace this path to where you stored your model
inferrence_param[
    "model_path"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/repos/public/deepinterpolation_models/deep_interpolation_neuropixel_v1/2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"

# Replace this path to where you want to store your output file
inferrence_param[
    "output_file"
] = "/Users/jeromel/test/ephys_tiny_continuous_deep_interpolation.h5"

jobdir = "/Users/jeromel/test/"

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
