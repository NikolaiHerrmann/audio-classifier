from visualization import *
from datasets import *
from utils import *
from models import *
from sklearn.model_selection import train_test_split

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
    history, cnn_model = train_cnn(input_shape_vowels, X_train, y_train, n_classes=9)
    if plot:
        plot_cnn_training(history)
    eval_cnn(cnn_model, X_test_vowels_uni, y_test_vowels)

    # Cross-validation
    cross_val_cnn(input_shape_vowels, X_train, y_train, num_folds=5, n_classes=9)

    # RF
    X_feat_train_vowels = extract_features(X_train_vowels_uni)
    X_feat_test_vowels = extract_features(X_test_vowels_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_vowels, y_train_vowels]))
    if plot:
        plot_tsne(X_feat_train_vowels, y_train_vowels)
        plot_tsne(X_feat_test_vowels, y_test_vowels)

    # One model
    model = train_rf(X_train, y_train)
    cm, acc, f1 = eval_rf(model, X_feat_test_vowels, y_test_vowels)
    if plot:
        plot_rf_training(cm)

    # Cross-validation
    cross_val_rf(X_train, y_train, 5)

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
    history_digits, cnn_model_digits = train_cnn(input_shape_digits, X_train, y_train, n_classes=6)
    if plot:
        plot_cnn_training(history_digits)
    eval_cnn(cnn_model_digits, X_test_digits_uni, y_test_digits)

    # Cross-validation
    cross_val_cnn(input_shape_digits, X_train, y_train, num_folds=5, n_classes=6)

    X_feat_train_digits = extract_features(X_train_digits_uni, n_windows=5)
    X_feat_test_digits = extract_features(X_test_digits_uni, n_windows=5)
    X_train, y_train = tuple(map(np.array, [X_feat_train_digits, y_train_digits]))

    # One model
    model = train_rf(X_train, y_train)
    cm, acc, f1 = eval_rf(model, X_feat_test_digits, y_test_digits)
    if plot:
        plot_rf_training(cm)

    # Cross-validation
    cross_val_rf(X_train, y_train, 5)

if __name__ == "__main__":
    train_japanese_vowels()
    train_spoken_digits()
