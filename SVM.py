from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

def SVM(csv_path):
    test_data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    x_train = np.loadtxt(r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\finaltest8\data10.csv',delimiter=',', dtype=np.float32)
    y_train = np.loadtxt(r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\finaltest8\data10.csv',delimiter=',', dtype=np.float32)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()     # 모든 특성들이 0과 1사이에 위치하도록 데이터를 비례적으로 조정
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    x_test = []
    x_test.append(test_data[len(test_data) - 1])
    x_test = np.array(x_test)

    x_test = x_test[:, 0:20]

    estimator = SVC(kernel='rbf', C=3, gamma=0.22)
    estimator.fit(x_train, y_train)


    y_predict = estimator.predict(x_test)

    return y_predict