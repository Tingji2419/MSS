import numpy as np
from generate_series import Artifical_Signal
from utils import set_seed
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb

from numpy.lib.stride_tricks import sliding_window_view


set_seed(42)

def pre_train(num=100000, windows_size=200):
    dataset = Artifical_Signal(num=num, windows_size=windows_size)
    X, y = dataset.generate_signal()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    src = MinMaxScaler()
    X_train = src.fit_transform(X_train)
    X_test = src.fit_transform(X_test)
    print('Totoal:', len(y))
    print('1 in label:', sum(y) / len(y))
    # classfier = xgb.XGBClassifier(max_depth=3, min_child_weight=2, n_estimators=120, gamma=0.5)
    classfier = xgb.XGBClassifier()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    # classfier.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
    classfier.fit(X_train, y_train)
    return classfier

def probe(classfier, ts, ws=200):
    if classfier == None:
        classfier = pre_train()
    ts_split = sliding_window_view(ts, ws)
    predict = np.array(classfier.predict(ts_split)) / (ws - 1)
    # split_num = len(ts) // ws
    # ts_split = ts[: (split_num * ws)].reshape(-1, ws)
    # predict = np.array(classfier.predict(ts_split))
    return int(sum(predict))