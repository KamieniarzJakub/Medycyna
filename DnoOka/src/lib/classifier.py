import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

def prepare_dataset(features: np.ndarray, labels: np.ndarray):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(features, labels)
    return X_res, y_res 

def train_classifier(X: np.ndarray, y: np.ndarray):
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    return classification_report(y_test, preds, output_dict=True)
