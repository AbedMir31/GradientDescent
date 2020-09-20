import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LEARNING_RATE = .01
EPOCH = 10000


def grad_descent(x, y, w, learning_rate, epoch):
    parameterlog = []
    for ep in range(0, epoch):
        pred = x * w
        error = pred - y
        mse = calculatemse(len(x), pred, y)
        gradient = x.T.dot(error) / x.shape[0]
        w += -learning_rate * gradient
        print('epoch: %d\tmse: %.2f' % (ep, mse))
        parameterlog.append(mse)
    return w, parameterlog


def calculatemse(n, h, y):
    return (1/(2*n)) * (np.sum(h - y) ** 2)


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/AbedMir31/GradientDescent/master/hour.csv'
    data = pd.read_csv(url, header=1)

    totalX = data.iloc[0:1000, 10]  # X = Temperature Column
    totalY = data.iloc[0:1000, 16]  # Y = Biker Count

    X_train, X_test = train_test_split(totalX, test_size=.2, random_state=69)
    Y_train, Y_test = train_test_split(totalY, test_size=.2, random_state=69)

    weights = np.random.uniform(low=0.0, high=1.0, size=(len(X_train)))

    # TRAIN
    new_weight, oldlog = grad_descent(X_train, Y_train, weights, LEARNING_RATE, EPOCH)
    log = open('log.txt', 'a')
    log.write('training data:\tstarting mse: %.2f\tending mse: %.2f\n' % (oldlog[0], oldlog[-1]))

    # TEST
    new_weight, newlog = grad_descent(X_train, Y_train, new_weight, LEARNING_RATE, EPOCH)
    log = open('log.txt', 'a')
    log.write('test data:\t starting mse: %f\tending mse: %.2f\n' % (newlog[0], newlog[-1]))

