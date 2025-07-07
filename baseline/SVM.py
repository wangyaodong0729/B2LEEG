import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
# def load_data(path):
#     data = loadmat(path)
#     X = data['features']
#     y = data['classes'].squeeze()  # 确保标签是一维数组
#     return X, y
# def load_data(path):
#     data = loadmat(path)
#     features = torch.tensor(data['features'], dtype=torch.float32)
#     labels = torch.tensor(data['classes'], dtype=torch.float32)
#     return features, labels

def main():
    df = pd.read_csv('/home/wyd/spikebls/pdoc.csv')
    features = df.iloc[1:, :-10].to_numpy()
    labels = df.iloc[1:, -1].to_numpy()
# def main():
#     # 加载数据
#     X, y = load_data('/home/wyd/spikebls/sleepS01features.mat')

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    # 特征标准化
    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)60.
    X_test = scaler.transform(X_test)

    # 创建SVM模型
    model = SVC(kernel='linear')  # 可以选择不同的核函数，如 'linear', 'poly', 'rbf'
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    main()
