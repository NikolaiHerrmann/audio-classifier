import os

# Note: Set environment variables before importing Keras/TensorFlow

# Force targetting CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from visualization import *
from datasets import *
from utils import *
from handcrafted import HandcraftedModel
from cnn import CnnModel

def train_japanese_vowels(plot=False):
    # Data preparation
    X_train_vowels, y_train_vowels, X_test_vowels, y_test_vowels = get_japanese_vowels()
    rec_len = 15
    X_train_vowels_uni, X_test_vowels_uni = pre_process(X_train_vowels, X_test_vowels, rec_len)
    input_shape_vowels = (rec_len, 12)
    X_train, y_train = tuple(map(np.array, [X_train_vowels_uni, y_train_vowels]))

    ######################## CNN ########################
    print("######## CNN MODEL ########")
    cnn_model = CnnModel(input_shape=input_shape_vowels, n_classes=9, data="vowels")
    history, model = cnn_model.train(X_train, y_train)

    # Cross-validation
    history = cnn_model.cross_val(X_train, y_train, num_folds=5)
    if plot:
        plot_cross_val(history, "Vowels")

    # Final accuracy CNN
    cm, accuracy, f1 = cnn_model.eval(model, X_test_vowels_uni, y_test_vowels)
    print(f'Test F1: {f1} / Test accuracy: {accuracy}')
    if plot:
        plot_cm(cm, "Vowels")

    ######################## HC ########################
    print("######## HC MODEL ########")
    X_feat_train_vowels = extract_features(X_train_vowels_uni)
    X_feat_test_vowels = extract_features(X_test_vowels_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_vowels, y_train_vowels]))

    hand_crafted_model = HandcraftedModel('svm', 'vowels')
    model = hand_crafted_model.train(X_train, y_train)

    # Cross-validation
    hand_crafted_model.cross_val(X_train, y_train, 5)

    cm, acc, f1 = hand_crafted_model.eval(model, X_feat_test_vowels, y_test_vowels)
    print(f'Test accuracy: {acc} / Test F1: {f1}')
    if plot:
        plot_cm(cm, "Vowels", "SVM")


def train_spoken_digits(plot=False):
    # Data preparation
    X_train_digits, y_train_digits, X_test_digits, y_test_digits = get_spoken_digits()
    rec_len = 27
    X_train_digits_uni, X_test_digits_uni = pre_process(X_train_digits, X_test_digits, rec_len)
    input_shape_digits = (rec_len, 12)
    X_train, y_train = tuple(map(np.array, [X_train_digits_uni, y_train_digits]))

    ######################## CNN ########################
    print("######## CNN MODEL ########")
    cnn_model = CnnModel(input_shape=input_shape_digits, n_classes=6, data="digits")
    history_digits, cnn_model_digits = cnn_model.train(X_train, y_train)

    # Cross-validation
    history = cnn_model.cross_val(X_train, y_train, num_folds=5)
    if plot:
        plot_cross_val(history, "Digits")
    
    # Final accuracy CNN
    cm, accuracy, f1 = cnn_model.eval(cnn_model_digits, X_test_digits_uni, y_test_digits)
    print(f'Test F1: {f1} / Test accuracy: {accuracy}')
    if plot:
        plot_cm(cm, "Digits")

    ######################## HC ########################
    print("######## HC MODEL ########")
    X_feat_train_digits = extract_features(X_train_digits_uni)
    X_feat_test_digits = extract_features(X_test_digits_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_digits, y_train_digits]))

    hand_crafted_model = HandcraftedModel('svm', 'digits')
    model = hand_crafted_model.train(X_train, y_train)

    # Cross-validation
    hand_crafted_model.cross_val(X_train, y_train, 5)

    cm, acc, f1 = hand_crafted_model.eval(model, X_feat_test_digits, y_test_digits)
    print(f'Test accuracy: {acc} / Test F1: {f1}')
    if plot:
        plot_cm(cm, "Digits", "SVM")


if __name__ == "__main__":
    print("######## VOWELS DATASET ########")
    train_japanese_vowels(True)
    print()
    print("######## DIGITS DATASET ########")
    train_spoken_digits(True)
