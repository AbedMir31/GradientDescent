import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/AbedMir31/GradientDescent/master/hour.csv'
    input = pd.read_csv(url, header=1)

    X = np.array(input.iloc[0:1000, 10]).reshape(-1, 1)  # X = Temperature Column
    Y = np.array(input.iloc[0:1000, 16]).reshape(len(X), )  # Y = Biker Count

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=69)

    regr = SGDRegressor(max_iter=10000)
    regr.fit(X_train, Y_train)

    y_pred = regr.predict(X_test)
    log = open('log2.txt', 'a')
    log.write('MSE: %.2f\t R2: %.2f\n' % (mean_squared_error(Y_test, y_pred), r2_score(Y_test, y_pred)))
    log.close()
    plt.scatter(X_test, Y_test)
    plt.xlabel = 'Temperature'
    plt.ylabel = 'Count'
    plt.plot(X_test, y_pred)
    plt.show()


