import multiprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from tqdm import tqdm
import keras_tuner as kt
from utils import RANDOM_STATE

class CnnModel:
    def __init__(self, input_shape, n_classes, hp_optimization=False, data=None) -> None:
        self.input_shape = input_shape
        self.n_classes = n_classes
        params = {
            "vowels": [96, 5],
            "digits": [128, 2]
        }
        self.best_params = {
            "filters": params[data][0],
            "learning_rate": 0.01,
            "kernel_size": params[data][1]
        } if not hp_optimization else None
    
    def get_model(self):
        """
        Returns a compiled model with fixed hyper-parameters.
        """
        self.learning_rate = float(self.best_params.get('learning_rate'))
        self.filters = int(self.best_params.get('filters'))
        self.kernel_size = int(self.best_params.get('kernel_size'))
        model = Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(self.n_classes, activation="softmax")
            ]
        )

        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                        loss="sparse_categorical_crossentropy",
                        metrics=["sparse_categorical_accuracy"],
        )
        return model
        
    def model_builder(self, hp):
        """
        Builds a model for KerasTuner to tune.
        """
        hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
        hp_kernels = hp.Int('kernel_size', min_value=1, max_value=5, step=1)
        model = Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.Conv1D(filters=hp_filters, kernel_size=hp_kernels, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(self.n_classes, activation="softmax")
            ]
        )
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
        )
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, valid=[]):
        """
        Trains the CNN model by optionally performing tuning.
        """
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        if not self.best_params:
            self.tuner = kt.BayesianOptimization(
                self.model_builder,
                objective='val_accuracy',
                seed=RANDOM_STATE,
                overwrite=True
            )
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            self.tuner.search(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=0, callbacks=[stop_early])
            self.best_params = self.tuner.get_best_hyperparameters(num_trials=1)[0]
            print(f"learning rate: {self.best_params.get('learning_rate')}")
            print(f"filters: {self.best_params.get('filters')}")
            print(f"kernel size: {self.best_params.get('kernel_size')}")
        cnn_model = self.get_model()
        history = cnn_model.fit(
            train_dataset,
            validation_data=valid,
            epochs=epochs,
            verbose=0,
            workers=multiprocessing.cpu_count()
        )
        return history, cnn_model

    def cross_val(self, X_train, y_train, num_folds):
        """
        Performs cross-validation and saves results.
        """
        acc_per_fold = []
        f1_per_fold = []
        history_per_fold = []
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
        for train, test in tqdm(kfold.split(X_train, y_train), total=num_folds):
            valid_dataset = tf.data.Dataset.from_tensor_slices((X_train[test], y_train[test])).batch(32)
            history, cnn_model = self.train(X_train[train], y_train[train], valid=valid_dataset)
            _, acc, f1 = self.eval(cnn_model, X_train[test], y_train[test])
            f1_per_fold.append(f1)
            acc_per_fold.append(acc)
            history_per_fold.append(history)
        print(f"Cross-validation results for {num_folds} folds -> avg. f1: {sum(f1_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")
        return history_per_fold

    def eval(self, model, X_test, y_test):
        """
        Evaluates the model on a test set.
        """
        X_test, y_test = tuple(map(np.array, [X_test, y_test]))
        y_pred = model.predict(X_test)
        y_pred = [np.argmax(i) for i in y_pred]
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return cm, acc, f1