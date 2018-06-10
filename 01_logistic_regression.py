import numpy as np

'''
データの生成
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

'''
モデル設定
'''
w = np.zeros(2)
b = .0
learning_rate = 0.1


def y(x):
    return sigmoid(x)  # WRITE ME!


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


'''
モデル学習
'''
for epoch in range(200):
    delta_w = .0
    delta_b = .0

    for i in range(len(X)):
        preds = y(X[i])
        t = Y[i][0]

        delta_w += (t - preds) * X[i] # WRITE ME!
        delta_b += (t - preds) # WRITE ME!

    w += learning_rate * delta_w
    b += learning_rate * delta_b

'''
学習結果の確認
'''
print('output probability:')
for _, x in enumerate(X):
    print(y(x))
