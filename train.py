from visualization import *
from datasets import *
from utils import *
from handcrafted import HandcraftedModel
from cnn import CnnModel

def train_japanese_vowels(plot=False):
    X_train_vowels, y_train_vowels, X_test_vowels, y_test_vowels = get_japanese_vowels()
    rec_len = 15
    X_train_vowels_uni, X_test_vowels_uni = pre_process(X_train_vowels, X_test_vowels, rec_len)
    input_shape_vowels = (rec_len, 12)
    X_train, y_train = tuple(map(np.array, [X_train_vowels_uni, y_train_vowels]))

    # One model
    cnn_model = CnnModel(input_shape=input_shape_vowels, n_classes=9)
    history, model = cnn_model.train(X_train, y_train)
    if plot:
        plot_cnn_training(history)
    # Cross-validation
    history = cnn_model.cross_val(X_train, y_train, num_folds=5)
    if plot:
        plot_cross_val(history)

    # Final accuracy CNN
    loss, accuracy = cnn_model.eval(model, X_test_vowels_uni, y_test_vowels)
    print(f'Test loss: {loss} / Test accuracy: {accuracy}')

    # RF
    X_feat_train_vowels = extract_features(X_train_vowels_uni)
    X_feat_test_vowels = extract_features(X_test_vowels_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_vowels, y_train_vowels]))
    if plot:
        plot_tsne(X_feat_train_vowels, y_train_vowels)
        plot_tsne(X_feat_test_vowels, y_test_vowels)

    # One model
    hand_crafted_model = HandcraftedModel('svm')
    model = hand_crafted_model.train(X_train, y_train)
    cm, acc, f1 = hand_crafted_model.eval(model, X_feat_test_vowels, y_test_vowels)
    if plot:
        plot_rf_training(cm)

    # Cross-validation
    hand_crafted_model.cross_val(X_train, y_train, 5)
    
    print(f'Test accuracy: {acc} / Test F1: {f1}')

def train_spoken_digits(plot=False):
    X_train_digits, y_train_digits, X_test_digits, y_test_digits = get_spoken_digits()
    rec_len = 27
    X_train_digits_uni, X_test_digits_uni = pre_process(X_train_digits, X_test_digits, rec_len)
    input_shape_digits = (rec_len, 12)
    X_train, y_train = tuple(map(np.array, [X_train_digits_uni, y_train_digits]))

    # One model
    cnn_model = CnnModel(input_shape=input_shape_digits, n_classes=6, hp_optimization=True)
    history_digits, cnn_model_digits = cnn_model.train(X_train, y_train)
    if plot:
        plot_cnn_training(history_digits)

    # Cross-validation
    history = cnn_model.cross_val(X_train, y_train, num_folds=5)
    if plot:
        plot_cross_val(history)
    
    # Final accuracy CNN
    loss, accuracy = cnn_model.eval(cnn_model_digits, X_test_digits_uni, y_test_digits)
    print(f'Test loss: {loss} / Test accuracy: {accuracy}')

    X_feat_train_digits = extract_features(X_train_digits_uni)
    X_feat_test_digits = extract_features(X_test_digits_uni)
    X_train, y_train = tuple(map(np.array, [X_feat_train_digits, y_train_digits]))

    # One model
    hand_crafted_model = HandcraftedModel('svm')
    model = hand_crafted_model.train(X_train, y_train)
    cm, acc, f1 = hand_crafted_model.eval(model, X_feat_test_digits, y_test_digits)
    if plot:
        plot_rf_training(cm)

    # Cross-validation
    hand_crafted_model.cross_val(X_train, y_train, 5)
    print(f'Test accuracy: {acc} / Test F1: {f1}')

if __name__ == "__main__":
    train_japanese_vowels()
    train_spoken_digits()
