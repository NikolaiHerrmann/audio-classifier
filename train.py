from visualization import *
from datasets import *
from utils import *
from models import *
from sklearn.model_selection import train_test_split
from cnn import CnnModel

def train_japanese_vowels(plot=False):
    X_train_vowels, y_train_vowels, X_test_vowels, y_test_vowels = get_japanese_vowels()
    if plot:
        plot_recordings(X_train_vowels)
        plot_rec_len_freq(X_train_vowels, "Train")
        plot_rec_len_freq(X_test_vowels, "Test")

    X_train_vowels_uni, X_test_vowels_uni = pre_process(X_train_vowels, X_test_vowels, rec_len=30)
    input_shape_vowels = (30, 12)
    X_train, y_train = tuple(map(np.array, [X_train_vowels_uni, y_train_vowels]))

    # One model
    cnn_model = CnnModel(input_shape=input_shape_vowels, n_classes=9)
    history, model = cnn_model.train(X_train, y_train)
    if plot:
        plot_cnn_training(history)
    loss, accuracy = cnn_model.eval(model, X_test_vowels_uni, y_test_vowels)
    print(f'Test loss: {loss} / Test accuracy: {accuracy}')

    # Cross-validation
    cnn_model.cross_val(X_train, y_train, num_folds=5)

    # RF
    X_feat_train_vowels = extract_features(X_train_vowels_uni)
    X_feat_test_vowels = extract_features(X_test_vowels_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_vowels, y_train_vowels]))
    if plot:
        plot_tsne(X_feat_train_vowels, y_train_vowels)
        plot_tsne(X_feat_test_vowels, y_test_vowels)

    # One model
    model = train_handcrafted(X_train, y_train)
    cm, acc, f1 = eval_rf(model, X_feat_test_vowels, y_test_vowels)
    if plot:
        plot_rf_training(cm)
    print(f'Test accuracy: {acc} / Test F1: {f1}')

    # Cross-validation
    cross_val_handcrafted(X_train, y_train, 5)

def train_spoken_digits(plot=False):
    X_digits, y_digits_num, y_digits_speaker = get_spoken_digits()
    X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits_speaker, test_size=0.5, stratify=y_digits_speaker)
    if plot:
        plot_rec_len_freq(X_train_digits, "Train", xmax=160)
        plot_rec_len_freq(X_test_digits, "Test", xmax=80)

    X_train_digits_uni, X_test_digits_uni = pre_process(X_train_digits, X_test_digits, 50)
    input_shape_digits = (50, 12)
    X_train, y_train = tuple(map(np.array, [X_train_digits_uni, y_train_digits]))

    # One model
    cnn_model = CnnModel(input_shape=input_shape_digits, n_classes=6)
    history_digits, cnn_model_digits = cnn_model.train(X_train, y_train)
    if plot:
        plot_cnn_training(history_digits)
    loss, accuracy = cnn_model.eval(cnn_model_digits, X_test_digits_uni, y_test_digits)
    print(f'Test loss: {loss} / Test accuracy: {accuracy}')

    # Cross-validation
    cnn_model.cross_val(X_train, y_train, num_folds=5)

    X_feat_train_digits = extract_features(X_train_digits_uni, n_windows=5)
    X_feat_test_digits = extract_features(X_test_digits_uni, n_windows=5)
    X_train, y_train = tuple(map(np.array, [X_feat_train_digits, y_train_digits]))

    # One model
    model = train_handcrafted(X_train, y_train)
    cm, acc, f1 = eval_rf(model, X_feat_test_digits, y_test_digits)
    if plot:
        plot_rf_training(cm)
    print(f'Test accuracy: {acc} / Test F1: {f1}')

    # Cross-validation
    cross_val_handcrafted(X_train, y_train, 5)

if __name__ == "__main__":
    train_japanese_vowels()
    train_spoken_digits()
