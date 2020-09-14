import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

LEARNING_RATE = .0001
NUM_ITERATIONS = 1000


def grad_descent(x, y, learning_rate, num_iterations, dataset_type):
    log = open("log.txt", "a")
    m = 0
    c = 0
    n = float(len(x))
    for i in range(num_iterations):
        y_prime = m * x + c
        m_prime = (-2 / n) * sum(x * (y - y_prime))
        c_prime = (-2 / n) * sum(y - y_prime)
        m = m - learning_rate * m_prime
        c = c - learning_rate * c_prime
    log.write('LOG: dataset type: ' + dataset_type + ',\t')
    log.write('learning rate: ' + str(learning_rate) + ',\t')
    log.write('num_iterations: ' + str(num_iterations) + '\n')
    return m * x + c


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/AbedMir31/GradientDescent/master/leaf.csv'
    data = pd.read_csv(url)
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=69)

    y_pred = grad_descent(X_train, Y_train, LEARNING_RATE, NUM_ITERATIONS, "train")
    plt.scatter(X_test, Y_test)
    plt.plot([min(X_test), max(X_test)], [min(y_pred), max(y_pred)], color='red')
    plt.show()
