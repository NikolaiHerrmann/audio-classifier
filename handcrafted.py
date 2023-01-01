from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils import RANDOM_STATE
from tqdm import tqdm

class HandcraftedModel:
    def __init__(self, type_) -> None:
        types_ = {
            "rf": (RandomForestClassifier(random_state=RANDOM_STATE), {'n_estimators': [10, 20, 50, 100]}),
            "lr": (LogisticRegression(random_state=RANDOM_STATE), {'C': [1, 10, 20, 50], 'solver': ['liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}),
            "svm": (SVC(random_state=RANDOM_STATE), {'C': [1, 10, 20, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [3, 6, 9]})
        }
        self.model, self.params = types_[type_]
        
    def train(self, X_train, y_train):
        model = GridSearchCV(self.model, self.params, n_jobs=-1)
        X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train, random_state=RANDOM_STATE)
        model.fit(X_train_shuffle, y_train_shuffle)

        # print("\n The best estimator across ALL searched params:\n", model.best_estimator_)
        # print("\n The best score across ALL searched params:\n", model.best_score_)
        # print("\n The best parameters across ALL searched params:\n", model.best_params_)
        return model


    def cross_val(self, X_train, y_train, num_folds):
        acc_per_fold = []
        f1_per_fold = []
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
        for train, test in tqdm(kfold.split(X_train, y_train), total=5):
            model = self.train(X_train[train], y_train[train])
            cm, acc, f1 = self.eval(model, X_train[test], y_train[test])
            acc_per_fold.append(acc)
            f1_per_fold.append(f1)
        print(f"Cross-validation results for {num_folds} folds -> avg. f1: {sum(f1_per_fold) / num_folds}, avg. accuracy: {sum(acc_per_fold) / num_folds}")

    def eval(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return confusion_matrix(y_test, y_pred), acc, f1

