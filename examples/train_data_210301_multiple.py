import os, sys
FILE_PATH = os.path.realpath(__file__)
EXAMPLES_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(EXAMPLES_PATH)
sys.path.append(MODULE_PATH)

import deepinterpolation as de
# import sys
from shutil import copyfile
# import os
from deepinterpolation.generic import JsonSaver, ClassLoader
from deepinterpolation.generator_collection import CollectorGenerator
import datetime
from typing import Any, Dict
import pathlib
from copy import deepcopy

# This is used for record-keeping
now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

# Initialize meta-parameters objects
training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

# An epoch is defined as the number of batches pulled from the dataset. Because our datasets are VERY large. Often, we cannot
# go through the entirity of the data so we define an epoch slightly differently than is usual.
steps_per_epoch = 5  # 5

# Those are parameters used for the Validation test generator. Here the test is done on the beginning of the data but
# this can be a separate file
generator_test_param["type"] = "generator"  # type of collection
generator_test_param["name"] = "SingleTifGeneratorRandomX"  # Name of object in the collection
generator_test_param[
    "pre_post_frame"
] = 30  # 30  # Number of frame provided before and after the predicted frame
generator_test_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "sample_data"
)  # only specify base path here.
generator_test_param["batch_size"] = 4  # 5
generator_test_param["start_frame"] = 3800  # 1900
generator_test_param["end_frame"] = -1
generator_test_param[
    "pre_post_omission"
] = 0  # Number of frame omitted before and after the predicted frame
generator_test_param["steps_per_epoch"] = steps_per_epoch

# Those are parameters used for the main data generator
generator_param["type"] = "generator"
generator_param["steps_per_epoch"] = steps_per_epoch
generator_param["name"] = "SingleTifGeneratorRandomX"
generator_param["pre_post_frame"] = 30  # 30
generator_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "sample_data"
)  # only specify base path here
generator_param["batch_size"] = 4  # 5
generator_param["start_frame"] = 0
generator_param["end_frame"] = -1  # -1
generator_param["N_train"] = 1000
generator_param["pre_post_omission"] = 0

# Those are parameters used for the network topology
network_param["type"] = "network"
network_param[
    "name"
] = "unet_single_1024"  # Name of network topology in the collection

# Those are parameters used for the training process
training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param[
    "period_save"
] = 25  # network model is potentially saved during training between a regular nb epochs
training_param["nb_gpus"] = 1  # 0
training_param["apply_learning_decay"] = 0
training_param[
    "nb_times_through_data"
] = 1  # if you want to cycle through the entire data. Two many iterations will cause noise overfitting
training_param["learning_rate"] = 0.0001
training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
training_param["loss"] = "mean_absolute_error"
training_param[
    "nb_workers"
] = 16  # this is to enable multiple threads for data generator loading. Useful when this is slower than training

training_param["model_string"] = (
    "210301_multi_"
    +
    network_param["name"]
    + "_"
    + training_param["loss"]
    + "_"
    + training_param["run_uid"]
)

# Where do you store ongoing training progress
jobdir = os.path.join(
    "/home/jbraun/bin/deepinterpolation/runs", training_param["model_string"] + "_" + run_uid,
)
training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

# Here we create all json files that are fed to the training. This is used for recording purposes as well as input to the
# training process
path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

# "longterm_003_crop.tif"
dataset_names = ["210301_001_crop.tif", "210301_002_crop.tif"]
train_paths = [os.path.join(generator_param["train_path"], dataset_name) for dataset_name in dataset_names]

path_generators = []
path_test_generators = []
for i_ds, train_path in enumerate(train_paths):
    path_generators.append(os.path.join(jobdir, "generator_{}.json".format(i_ds)))
    generator_param["train_path"] = train_path
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generators[i_ds])

    path_test_generators.append(os.path.join(jobdir, "test_generator_{}.json".format(i_ds)))
    generator_test_param["train_path"] = train_path
    json_obj = JsonSaver(generator_test_param)
    json_obj.save_json(path_test_generators[i_ds])

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

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
