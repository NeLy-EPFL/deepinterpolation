import os
import numpy as np
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import deepinterpolation.loss_collection as lc
from tensorflow.keras.layers import Input
from deepinterpolation.generic import JsonLoader
import math
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model


def create_decay_callback(initial_learning_rate, epochs_drop):
    """ This is a helper function to return a configured
    learning rate decay callback. It uses passed parameters
    to define the behavior of the decay
    """

    def step_decay(epoch):
        """ learning decay callback"""

        drop = 0.5
        decrease_pow = math.floor((1 + epoch) / epochs_drop)
        lrate = initial_learning_rate * math.pow(drop, decrease_pow)

        return lrate

    return step_decay


class core_trainer:
    # This is the generic trainer class
    # auto_compile can be set to False when doing an hyperparameter search to allow modification of the model
    def __init__(
        self,
        generator_obj,
        test_generator_obj,
        network_obj,
        trainer_json_path,
        auto_compile=True,
    ):

        self.network_obj = network_obj
        self.local_generator = generator_obj
        self.local_test_generator = test_generator_obj

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.
        json_obj.set_default("apply_learning_decay", 0)

        json_data = json_obj.json_data
        self.output_dir = json_data["output_dir"]
        self.run_uid = json_data["run_uid"]
        self.model_string = json_data["model_string"]
        self.batch_size = json_data["batch_size"]
        self.steps_per_epoch = json_data["steps_per_epoch"]
        self.loss_type = json_data["loss"]
        self.nb_gpus = json_data["nb_gpus"]
        self.period_save = json_data["period_save"]
        self.learning_rate = json_data["learning_rate"]

        if "nb_workers" in json_data.keys():
            self.workers = json_data["nb_workers"]
        else:
            self.workers = 16

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = json_data["initial_learning_rate"]
            self.epochs_drop = json_data["epochs_drop"]

        self.nb_times_through_data = json_data["nb_times_through_data"]

        # Generator has to be initialized first to provide
        # input size of network
        self.initialize_generator()

        if self.nb_gpus > 1:
            mirrored_strategy = tensorflow.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.initialize_network()

                self.initialize_callbacks()

                self.initialize_loss()

                self.initialize_optimizer()
                if auto_compile:
                    self.compile()
        else:
            if auto_compile:
                self.initialize_network()

            self.initialize_callbacks()

            self.initialize_loss()

            self.initialize_optimizer()

            if auto_compile:
                self.compile()

    def compile(self):
        self.local_model.compile(
            loss=self.loss, optimizer=self.optimizer
        )  # , metrics=['mae'])

    def initialize_optimizer(self):
        self.optimizer = RMSprop(lr=self.learning_rate)

    def initialize_loss(self):
        self.loss = lc.loss_selector(self.loss_type)

    def initialize_callbacks(self):
        checkpoint_path = os.path.join(
            self.output_dir,
            self.run_uid + "_" + self.model_string + "-{epoch:04d}-{val_loss:.4f}.h5",
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
            period=self.period_save,
        )

        # Add on epoch_end callback

        epo_end = OnEpochEnd([self.local_generator.on_epoch_end])

        if self.apply_learning_decay == 1:
            step_decay_callback = create_decay_callback(
                self.initial_learning_rate, self.epochs_drop
            )

            lrate = LearningRateScheduler(step_decay_callback)
            callbacks_list = [checkpoint, lrate]
        else:
            callbacks_list = [checkpoint]

        if not (tensorflow.__version__ == "2.3.0"):
            callbacks_list.append(epo_end)

        self.callbacks_list = callbacks_list

    def initialize_generator(self):
        self.epochs = self.nb_times_through_data * int(
            np.floor(len(self.local_generator) / self.steps_per_epoch)
        )

    def initialize_network(self):
        local_size = self.local_generator.get_input_size()

        input_img = Input(shape=local_size)
        self.local_model = Model(input_img, self.network_obj(input_img))

    def cache_validation(self):
        # This is used to remove IO duplication, leverage memory for validation and
        # avoid deadlocks that happens when using keras.utils.Sequence as validation datasets

        input_example = self.local_test_generator.__getitem__(0)
        nb_object = int(len(self.local_test_generator))

        input_shape = list(input_example[0].shape)
        nb_samples = input_shape[0]

        input_shape[0] = input_shape[0] * nb_object

        output_shape = list(input_example[1].shape)
        output_shape[0] = output_shape[0] * nb_object

        cache_input = np.zeros(shape=input_shape, dtype=input_example[0].dtype)
        cache_output = np.zeros(shape=output_shape, dtype=input_example[1].dtype)

        for local_index in range(len(self.local_test_generator)):
            local_data = self.local_test_generator.__getitem__(local_index)
            cache_input[
                local_index * nb_samples : (local_index + 1) * nb_samples, :
            ] = local_data[0]
            cache_output[
                local_index * nb_samples : (local_index + 1) * nb_samples, :
            ] = local_data[1]

        self.local_test_generator = (cache_input, cache_output)

    def run(self):
        # we first cache the validation data
        self.cache_validation()

        self.model_train = self.local_model.fit(
            self.local_generator,
            validation_data=self.local_test_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            max_queue_size=32,
            workers=self.workers,
            shuffle=False,
            use_multiprocessing=True,
            callbacks=self.callbacks_list,
            initial_epoch=0,
        )

    def finalize(self):
        loss = self.model_train.history["loss"]
        val_loss = self.model_train.history["val_loss"]

        # save losses
        save_loss_path = os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "_loss.npy"
        )
        np.save(save_loss_path, loss)

        save_val_loss_path = os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "_val_loss.npy"
        )
        np.save(save_val_loss_path, val_loss)

        # save history
        save_hist_path = os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "_model.h5"
        )

        self.local_model.save(save_hist_path)
        print("Saved model to disk")
        h = plt.figure()
        plt.plot(loss, label="loss " + self.run_uid)
        plt.plot(val_loss, label="val_loss " + self.run_uid)

        plt.xlabel(
            "number of epochs ("
            + str(self.batch_size * self.steps_per_epoch)
            + " samples/epochs)"
        )
        plt.ylabel("training loss")
        plt.legend()
        save_hist_path = os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "_losses.png"
        )
        plt.savefig(save_hist_path)
        plt.close(h)


# This is a helper class to fix an issue in tensorflow 2.0.
# the on_epoch_end callback from sequences is not called.
class OnEpochEnd(tensorflow.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


class transfer_trainer(core_trainer):
    # This class is used to fine-tune a pre-trained model with additional data
    
    def __init__(
        self,
        generator_obj,
        test_generator_obj,
        trainer_json_path,
        auto_compile=True,
    ):

        # self.network_obj = network_obj
        self.local_generator = generator_obj
        self.local_test_generator = test_generator_obj

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.
        json_obj.set_default("apply_learning_decay", 0)

        json_data = json_obj.json_data
        self.model_path = json_data["model_path"]
        self.output_dir = json_data["output_dir"]
        self.run_uid = json_data["run_uid"]
        self.model_string = json_data["model_string"]
        self.batch_size = json_data["batch_size"]
        self.steps_per_epoch = json_data["steps_per_epoch"]
        self.loss_type = json_data["loss"]
        self.nb_gpus = json_data["nb_gpus"]
        self.period_save = json_data["period_save"]
        self.learning_rate = json_data["learning_rate"]

        if "nb_workers" in json_data.keys():
            self.workers = json_data["nb_workers"]
        else:
            self.workers = 16

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = json_data["initial_learning_rate"]
            self.epochs_drop = json_data["epochs_drop"]

        self.nb_times_through_data = json_data["nb_times_through_data"]

        # Generator has to be initialized first to provide
        # input size of network
        self.initialize_generator()

        if self.nb_gpus > 1:
            mirrored_strategy = tensorflow.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.initialize_network()

                self.initialize_callbacks()

                self.initialize_loss()

                self.initialize_optimizer()
                if auto_compile:
                    self.compile()
        else:
            if auto_compile:
                self.initialize_network()

            self.initialize_callbacks()

            self.initialize_loss()

            self.initialize_optimizer()
            
            if auto_compile:
                self.compile()

    def initialize_network(self):
        self.local_model = load_model(
            self.model_path,
            custom_objects={"annealed_loss": lc.loss_selector("annealed_loss")},
        )
    
    def initialize_loss(self):
        self.loss = lc.loss_selector(self.loss_type)
        
        # For transfer learning, knowing the baseline validation loss is important
        baseline_val_loss = self.local_model.evaluate(self.local_test_generator)
        
        # save init losses
        save_loss_path = os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "init_val_loss.npy"
        )
        np.save(save_loss_path, baseline_val_loss)
