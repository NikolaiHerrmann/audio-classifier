from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils import RANDOM_STATE
from tqdm import tqdm

def train_handcrafted(X_train, y_train):
    # params = {'n_estimators': [10, 20, 50, 100]}
    # model = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), params, n_jobs=-1)

    # params = {'C': [1, 10, 20, 50], 'solver': ['liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
    # model = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), params)
    
    params = {'C': [1, 10, 20, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [3, 6, 9]}
    model = GridSearchCV(SVC(random_state=RANDOM_STATE), params, n_jobs=-1)
    X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train, random_state=RANDOM_STATE)
    model.fit(X_train_shuffle, y_train_shuffle)

    # print("\n The best estimator across ALL searched params:\n", model.best_estimator_)
    # print("\n The best score across ALL searched params:\n", model.best_score_)
    # print("\n The best parameters across ALL searched params:\n", model.best_params_)
    return model


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

def eval_rf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return confusion_matrix(y_test, y_pred), acc, f1

