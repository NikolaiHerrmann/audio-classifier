import multiprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from tqdm import tqdm
import keras_tuner as kt
from utils import RANDOM_STATE
import os

# prevent tensorflow gpu warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CnnModel:
    def __init__(self, input_shape, n_classes) -> None:
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.best_params = None
    
    def get_model(self):
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
                        metrics=["sparse_categorical_accuracy"],
        )
        return model

    def train(self, X_train, y_train, epochs=200, batch_size=32):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        if not self.best_params:
            self.tuner = kt.Hyperband(
                self.model_builder,
                objective='sparse_categorical_accuracy',
                max_epochs=100,
                factor=3,
                overwrite=True
            )
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
            self.tuner.search(train_dataset, epochs=epochs, callbacks=[stop_early])
            self.best_params = self.tuner.get_best_hyperparameters(num_trials=1)[0]
            self.learning_rate = float(self.best_params.get('learning_rate'))
            self.filters = int(self.best_params.get('filters'))
            self.kernel_size = int(self.best_params.get('kernel_size'))
            print(f"learning rate: {self.best_params.get('learning_rate')}")
            print(f"filters: {self.best_params.get('filters')}")
            print(f"kernel size: {self.best_params.get('kernel_size')}")
        cnn_model = self.get_model()
        history = cnn_model.fit(
            train_dataset,
            epochs=epochs,
            verbose=0,
            workers=multiprocessing.cpu_count()
        )
        return history, cnn_model

    def cross_val(self, X_train, y_train, num_folds):
        acc_per_fold = []
        loss_per_fold = []
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
        for train, test in tqdm(kfold.split(X_train, y_train), total=num_folds):
            history, cnn_model = self.train(X_train[train], y_train[train])
            loss, acc = self.eval(cnn_model, X_train[test], y_train[test])
            loss_per_fold.append(loss)
            acc_per_fold.append(acc)
        print(f"Cross-validation results for {num_folds} folds -> avg. loss: {sum(loss_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")

    def eval(self, model, X_test, y_test):
        X_test, y_test = tuple(map(np.array, [X_test, y_test]))
        score = model.evaluate(x=X_test, y=y_test, verbose=0)
        loss, accuracy = score[:2]
        return loss, accuracy