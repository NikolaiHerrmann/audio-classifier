import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils import RANDOM_STATE

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from tqdm import tqdm
import keras_tuner as kt


def train_handcrafted(X_train, y_train):
    # params = {'n_estimators': [10, 20, 50, 100]}
    # model = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), params, n_jobs=-1)

    # params = {'C': [1, 10, 20, 50], 'solver': ['liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
    # model = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), params)
    
    params = {'C': [1, 10, 20, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [3, 6, 9]}
    model = GridSearchCV(SVC(random_state=RANDOM_STATE), params, n_jobs=-1)
    X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train, random_state=RANDOM_STATE)
    model.fit(X_train_shuffle, y_train_shuffle)

    print("\n The best estimator across ALL searched params:\n", model.best_estimator_)
    print("\n The best score across ALL searched params:\n", model.best_score_)
    print("\n The best parameters across ALL searched params:\n", model.best_params_)
    return model

def model_builder(hp):
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    hp_kernels = hp.Int('kernel_size', min_value=1, max_value=5, step=1)
    model = Sequential(
        [
            layers.Input(shape=(30,12)),
            layers.Conv1D(filters=hp_filters, kernel_size=hp_kernels, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(9, activation="softmax")
        ]
    )
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=["sparse_categorical_accuracy"],
    )
    return model

def train_cnn(input_shape, X_train, y_train, n_classes, epochs=50, batch_size=32):    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    tuner = kt.Hyperband(model_builder,
                        objective='sparse_categorical_accuracy',
                        max_epochs=10,
                        factor=3)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    tuner.search(train_dataset, epochs=50, callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"learning rate: {best_hps.get('learning_rate')}")
    print(f"filters: {best_hps.get('filters')}")
    print(f"kernel size: {best_hps.get('kernel_size')}")
    cnn_model = tuner.hypermodel.build(best_hps)
    history = cnn_model.fit(train_dataset,
                            epochs=epochs,
                            verbose=0,
                            workers=multiprocessing.cpu_count())
    return history, cnn_model

def cross_val_handcrafted(X_train, y_train, num_folds):
    i = 1
    acc_per_fold = []
    f1_per_fold = []
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    for train, test in tqdm(kfold.split(X_train, y_train)):
        model = train_handcrafted(X_train[train], y_train[train])
        cm, acc, f1 = eval_rf(model, X_train[test], y_train[test])
        acc_per_fold.append(acc)
        f1_per_fold.append(f1)
        i += 1
    print(f"Cross-validation results for {num_folds} folds -> avg. f1: {sum(f1_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")

def cross_val_cnn(input_shape, X_train, y_train, num_folds, n_classes):
    i = 1
    acc_per_fold = []
    loss_per_fold = []
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    for train, test in tqdm(kfold.split(X_train, y_train)):
        history, cnn_model = train_cnn(input_shape, X_train[train], y_train[train], n_classes)
        loss, acc = eval_cnn(cnn_model, X_train[test], y_train[test])
        loss_per_fold.append(loss)
        acc_per_fold.append(acc)
        i += 1
    print(f"Cross-validation results for {num_folds} folds -> avg. loss: {sum(loss_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")

def eval_cnn(model, X_test, y_test):
    X_test, y_test = tuple(map(np.array, [X_test, y_test]))
    score = model.evaluate(x=X_test, y=y_test, verbose=0)
    loss, accuracy = score[:2]
    return loss, accuracy

def eval_rf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return confusion_matrix(y_test, y_pred), acc, f1

