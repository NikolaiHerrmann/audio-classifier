import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from utils import RANDOM_STATE

import tensorflow as tf
from tensorflow.keras import layers, Sequential


def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=RANDOM_STATE)

    X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train, random_state=RANDOM_STATE)

    model.fit(X_train_shuffle, y_train_shuffle)

    y_pred = model.predict(X_test)
    print("Accuracy", accuracy_score(y_test, y_pred))
    print("F1-Score", f1_score(y_test, y_pred, average='weighted'))

    cm = confusion_matrix(y_test, y_pred)
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.xlabel("Predicted Speaker")
    plt.ylabel("True Speaker")
    plt.show()

def train_model(input_shape, X_train, y_train, epochs=200, batch_size=32):
    X_train, y_train = tuple(map(np.array, [X_train, y_train]))
    cnn_model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(9, activation="softmax")
        ]
    )

    cnn_model.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["sparse_categorical_accuracy"],
    )
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    history = cnn_model.fit(x=train_dataset,
                            epochs=epochs,
                            verbose=0,
                            workers=multiprocessing.cpu_count()
    )
    return history, cnn_model

def cross_val(input_shape, X_train, y_train, num_folds=1):
    i = 1
    acc_per_fold = []
    loss_per_fold = []
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    for train, test in kfold.split(X_train, y_train):
        history, cnn_model = train_model(input_shape, X_train[train], y_train[train])
        loss, acc = eval_model(cnn_model, X_train[test], y_train[test])
        loss_per_fold.append(loss)
        acc_per_fold.append(acc)
        i += 1
    print(f"Cross-validation results for {num_folds} folds -> avg. loss: {sum(loss_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")

def eval_model(model, X_test, y_test):
    X_test, y_test = tuple(map(np.array, [X_test, y_test]))
    score = model.evaluate(x=X_test, y=y_test, verbose=0)
    loss, accuracy = score[:2]
    print(f'Test loss: {loss} / Test accuracy: {accuracy}')
    return loss, accuracy
