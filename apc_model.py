# experiments.py

import os
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from tensorflow import keras
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from APC.utils import get_class_weights, get_results_df, oversample_batch_generator_GRU, batch_generator_GRU
from APC.models.GRU import GRU_model
from APC.models.GRU_D import create_grud_model, load_grud_model
from APC.models.APC import *

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





# Assuming utility functions and models are correctly imported from their respective modules
if __name__ == "__main__":
    dataset_path = 'datasets/physionet2012.pickle'
    save_model_path = 'saved_models/gru_model.h5'
    model_name = 'GRU_model_example'

    data_handler = DataHandler(dataset_path)
    data_handler.load_data()
    class_balance = data_handler.get_class_balance()

    config = {
        "x_length": 100,  # Example value
        "n_features": 20,  # Example value
        "n_aux": 5,  # Example value
        "n_classes": 3,  # Example value
        "n_neurons": 64,  # Example value
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "recurrent_dropout": 0.5,
        "loss_type": 'categorical_crossentropy',
        "epochs": 50,
        "batch_size": 32,
        "loss_to_monitor": 'val_loss',
        "monitor_mode": 'min'
    }

    class_weights = get_class_weights(class_balance)
    experiment = GRUExperiment(data_handler.data, class_balance, config, save_model_path, model_name)
    results = experiment.run(method="class_weights", class_weights=class_weights)
    print(results)
