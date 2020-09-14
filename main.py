import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn.model_selection import train_test_split

LEARNING_RATE = .001
EPOCH = 100000


def grad_descent(x, y, learning_rate, epoch):

    m, c = .2, .2
    n = float(len(x))
    log_mse = []

    for _ in range(epoch):
        y_prime = y - (m*x + c)
        m = m - learning_rate * (-2 * x.dot(y_prime).sum() / n)
        c = c - learning_rate * (-2 * y_prime.sum() / n)
        log_mse.append(mean_squared_error(y, (m * x + c)))

    return m, c, log_mse


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/AbedMir31/GradientDescent/master/hour.csv'
    data = pd.read_csv(url, header=1)

    totalX = data.iloc[0:1000, 10]  # X = Temperature Column
    totalY = data.iloc[0:1000, 16]  # Y = Biker Count

    """
    plt.scatter(X, Y)
    plt.xlabel('Temperature (C)', fontsize=12)
    plt.ylabel('Biker Count', fontsize=12)
    plt.show()
    """

    X_train, X_test = train_test_split(totalX, test_size=.2, random_state=69)
    Y_train, Y_test = train_test_split(totalY, test_size=.2, random_state=69)

    m, b, mse_log = grad_descent(X_train, Y_train, LEARNING_RATE, EPOCH)
    Y_pred = m*X_train + b

    log = open('log.txt', 'a')
    log.write('learning rate: ' + str(LEARNING_RATE) + '\t')
    log.write('epoch: ' + str(EPOCH) + '\t')
    log.write('trial beginning MSE: ' + "{:.5f}".format(mse_log[0]) + '\t\t')
    log.write('trial completion MSE: ' + "{:.5f}".format(mse_log[-1]) + '\n')

    #  y_pred = grad_descent(X_train, Y_train, LEARNING_RATE, NUM_ITERATIONS, "train")
    #  plt.scatter(X_test, Y_test)
    #  plt.plot([min(X_test), max(X_test)], [min(y_pred), max(y_pred)], color='red')
    #  plt.show()
