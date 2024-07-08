# experiments.py

import os
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from tensorflow import keras
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from APC.models import APC
from APC.utils import get_class_weights, get_results_df, oversample_batch_generator_GRU, batch_generator_GRU, \
    APC_batch_generator, APC_GRUD_batch_generator
from APC.models.GRU import GRU_model
from APC.models.APC import *
from APC.utils import masked_mse,auprc
from APC.models.GRU_D import create_grud_model, load_grud_model
from APC.models.APC import *
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import backend as K
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

class DataHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None

    def load_data(self):
        with open(self.dataset_path, 'rb') as handle:
            self.data = pickle.load(handle)

    def get_class_balance(self):
        classes_total_dataset = np.concatenate([self.data["train"]["y_classes"], self.data["val"]["y_classes"], self.data["test"]["y_classes"]])
        class_balance = Counter(classes_total_dataset)
        class_balance = {c: class_balance[c] / classes_total_dataset.shape[0] for c in class_balance}
        return class_balance

class GRUExperiment:
    def __init__(self, data, class_balance, config, save_model_path, model_name):
        self.data = data
        self.class_balance = class_balance
        self.config = config
        self.save_model_path = save_model_path
        self.model_name = model_name

    def run(self, method="class_weights", class_weights=None):
        K.clear_session()
        Path(self.save_model_path).mkdir(parents=True, exist_ok=True)

        # Create model
        model = GRU_model(
            x_length=self.config["x_length"],
            n_features=self.config["n_features"],
            n_aux=self.config["n_aux"],
            n_classes=self.config["n_classes"],
            n_neurons=self.config["n_neurons"],
            learning_rate=self.config["learning_rate"],
            dropout_rate=self.config["dropout_rate"],
            recurrent_dropout=self.config["recurrent_dropout"],
            loss_type=self.config["loss_type"]
        )

        checkpoint = ModelCheckpoint(self.save_model_path, monitor=self.config["loss_to_monitor"], verbose=0, save_best_only=True, mode=self.config["monitor_mode"])

        # Train model
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        if method == "class_weights":
            history = model.fit(
                [self.data["train"]["X_train"], self.data["train"]["X_aux"]], self.data["train"]["y"],
                validation_data=([self.data["val"]["X_val"], self.data["val"]["X_aux"]], self.data["val"]["y"]),
                epochs=self.config["epochs"],
                callbacks=[reduce_lr, checkpoint],
                batch_size=self.config["batch_size"],
                class_weight=class_weights,
                verbose=2
            )
        elif method == "oversample":
            print("using batch generator")
            steps_per_epoch = np.ceil(self.data["train"]["X_train"].shape[0] / self.config["batch_size"])
            validation_steps = np.ceil(self.data["val"]["X_val"].shape[0] / self.config["batch_size"])

            training_generator = oversample_batch_generator_GRU(self.data["train"], self.class_balance, self.config["epochs"], self.config["batch_size"])
            validation_generator = batch_generator_GRU(self.data["val"], self.config["batch_size"])

            history = model.fit_generator(
                generator=training_generator,
                validation_data=validation_generator,
                epochs=self.config["epochs"],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[reduce_lr, checkpoint],
                verbose=2
            )

        # Load best weights
        model.load_weights(self.save_model_path)

        # Get results
        yhat_test = model.predict([self.data["test"]["X_test"], self.data["test"]["X_aux"]], verbose=1)
        y_pred_test = [np.argmax(y, axis=None, out=None) for y in yhat_test]
        y_test_ = [np.argmax(y, axis=None, out=None) for y in self.data["test"]["y"]]

        results = get_results_df(y_test_, y_pred_test, self.data["test"]["y"], yhat_test, self.model_name, self.config["n_classes"])

        return results


class APCExperiment:
    def __init__(self, data, class_balance, config, save_model_path, model_name,encoder = "GRU"):
        self.data = data
        self.class_balance = class_balance
        self.config = config
        self.save_model_path = save_model_path
        self.model_name = model_name
        self.encoder =encoder

    def run(self, method="class_weights", class_weights=None, stop_APC_grad=False,step=1):

        K.clear_session()
        Path(self.save_model_path).mkdir(parents=True, exist_ok=True)

        save_model_path = os.path.join(self.save_model_path, "apc_model.keras")
        model = create_APC_classifier(self.config, self.encoder,stop_APC_grad)
        if self.config["pre_trained_weights"] != 0:
            model.load_weights(self.config["pre_trained_weights"])

        # train model
        # training steps
        train_steps_per_epoch = np.ceil(self.data["train"]["X"].shape[0] / self.config["batch_size"])
        # validation steps
        validation_steps = np.ceil(self.data["val"]["X"].shape[0] / self.config["batch_size"])

        # add checkpoint to save best model weights

        training_generator = None
        validation_generator = None
        if  self.encoder == "GRU":
            training_generator = APC_batch_generator(self.data["train"], self.config["time_shift"], self.config["batch_size"])
            validation_generator = APC_batch_generator(self.data["val"], self.config["time_shift"], self.config["batch_size"])
        elif self.encoder == "GRUD":
            training_generator = APC_GRUD_batch_generator(self.data["train"], self.config["time_shift"], self.config["batch_size"])
            validation_generator = APC_GRUD_batch_generator(self.data["train"], self.config["time_shift"], self.config["batch_size"])

        def weighted_generator(generator, class_weights):
            while True:
                (x_batch, x_aux_batch), (y_1_batch, y_batch) = next(generator)


                if y_batch is None:
                    raise ValueError("y_batch is None")
                sample_weights = np.array([class_weights[class_id] for class_id in np.argmax(y_batch, axis=1)])
                yield (x_batch, x_aux_batch), (y_1_batch, y_batch), sample_weights

        input_shape = (48, 74)
        aux_shape = (9,)
        output_shape = (48, 74)
        num_classes = 2
        output_signature = (
            (
                tf.TensorSpec(shape=(self.config["batch_size"], *input_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(self.config["batch_size"], *aux_shape), dtype=tf.float32)
            ),
            (
                tf.TensorSpec(shape=(self.config["batch_size"], *output_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(self.config["batch_size"], num_classes), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(self.config["batch_size"],), dtype=tf.float32)
        )

        weighted_training_dataset = tf.data.Dataset.from_generator(
            lambda: weighted_generator(training_generator, class_weights),
            output_signature=output_signature
        )

        weighted_val_dataset = tf.data.Dataset.from_generator(
            lambda: weighted_generator(validation_generator, class_weights),
            output_signature=output_signature
        )

        train_steps_per_epoch = int(train_steps_per_epoch)
        validation_steps = int(validation_steps)

        model.history = model.fit(
            x=weighted_training_dataset,
            epochs=self.config["epochs"],
            validation_data=weighted_val_dataset,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2

        )

        y_pred = None
        y_test = None
        if step == 1:
            # pretraining step
            model.save(save_model_path)
            return print("done training.")


        else:  # step 2 or 3
            # load best weights


            test_steps_per_epoch =int (np.ceil(self.data["test"]["X"].shape[0] / self.config["batch_size"]))
            print(test_steps_per_epoch)
            if self.encoder == "GRU":
                test_generator = APC_batch_generator(self.data["test"], self.config["time_shift"],
                                                         self.config["batch_size"])
                weighted_test_generator = tf.data.Dataset.from_generator(
                    lambda: weighted_generator(test_generator, class_weights),
                    output_signature=output_signature
                )

                _,y_pred = model.predict(weighted_test_generator, steps=test_steps_per_epoch)

            elif self.encoder == "GRUD":
                test_generator = APC_GRUD_batch_generator(self.data["test"], self.config["time_shift"], self.config["batch_size"])
                class_pred, actual_classes = [], []
                i = 0
                for batch in test_generator:
                    if i < (test_steps_per_epoch + 1):
                        actual_classes.append(batch[1])
                        r, c = model.predict_on_batch(batch[0])
                        class_pred.append(c)
                        i += 1
                    else:
                        break

                y_test = [x[1] for i, x in enumerate(actual_classes)]
                y_pred = np.vstack(class_pred)


            y_test = self.data["test"]["y"]
            y_test_classes = self.data["test"]["y_classes"]
            y_test_classes = list(y_test_classes)
            y_pred_classes = [np.argmax(y, axis=None, out=None) for y in y_pred][:len(y_test_classes)]
            model_name = self.encoder + "- APC"
            y_pred = y_pred[:len(y_test)]

            results = get_results_df(y_test_classes, y_pred_classes, y_test, y_pred, model_name, self.config["n_classes"])
            return results

# Assuming utility functions and models are correctly imported from their respective modules
if __name__ == "__main__":
    dataset_path = 'datasets/physionet2012.pickle'
    save_model_path = 'saved_models/apc_model.weights.h5'
    model_name = 'APC_model_example'

    data_handler = DataHandler(dataset_path)
    data_handler.load_data()

    # Get class balance
    class_balance = data_handler.get_class_balance()
    class_weights = get_class_weights(class_balance)

    # Experiment configurations
    config_apc = {
        "x_length": 48,
        "n_features": 74,
        "n_aux": 9,
        "n_classes": 2,
        "n_neurons": 64,
        "learning_rate": 0.01,
        "dropout_rate": 0.5,
        "recurrent_dropout": 0.5,
        "loss_type": "categorical_crossentropy",
        "loss_to_monitor": "val_loss",
        "monitor_mode": "min",
        "epochs": 100,
        "batch_size": 32,
        "pretrain_epochs": 5,
        "fine_tune_epochs": 50,
        "fine_tune_learning_rate": 0.0001,
        "recurrent_dropout_rate": 0.0,
        "aux_dim": 9,
        "l1": 1,
        "l2": 0,
        "l1_type": masked_mse,
        "l2_type": 'binary_crossentropy',
        "evaluation_metric": f1_score,
        'pre_trained_weights':0,

    }



    time_shift_factor = 1

    config_apc["time_shift"] = time_shift_factor
    data_handler.data["train"]["X"] =     data_handler.data["train"]["X_train"]
    data_handler.data["val"]["X"] =     data_handler.data["val"]["X_val"]
    data_handler.data["test"]["X"] =     data_handler.data["test"]["X_test"]
    # Run APC experiment
    experiment_apc = APCExperiment(data_handler.data, class_balance, config_apc, save_model_path, "APC",encoder="GRU")
    results_apc = experiment_apc.run(method="class_weights", class_weights=class_weights)
    print("APC Experiment Results:")
    print(results_apc)

    config_apc["l1"] = 0
    config_apc["l2"] = 1
    config_apc["learning_rate"] = 0.01
    config_apc["pre_trained_weights"] = r'/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/saved_models/apc_model.weights.h5/apc_model.keras'


    config_apc["loss_to_monitor"] = "val_dense_3_auprc"
    config_apc["monitor_mode"] = "max"
    APC_step2_weights_path = "apc_models/APC_step2_n{}_weights_{}".format(time_shift_factor, "GRU")
    APC_step2_results = APCExperiment(data=data_handler.data, class_balance=class_balance, config=config_apc
                                      ,model_name="APC",encoder="GRU",save_model_path=save_model_path)
    res =APC_step2_results.run(step=2, stop_APC_grad=True,method="class_weights",class_weights=class_weights)
    print(res)
