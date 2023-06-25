import numpy as np
from sklearn.linear_model import LinearRegression


def grad_des(x, y):
    m = c = 0
    iterations = 10000
    n = len(x)
    best_cost = float('inf')
    best_m = best_c = best_lr = best_iteration = 0

    for lr in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        for iteration in range(iterations):
            y_pred = m * x + c
            q = y - y_pred
            cost = (1 / n) * sum([val**2 for val in q])
            dm = -(2 / n) * sum(x * q)
            dc = -(2 / n) * sum(q)
            m = m - lr * dm
            c = c - lr * dc
            if cost < best_cost:
                best_cost = cost
                best_m, best_c, best_lr, best_iteration = m, c, lr, iteration

    print("Best values - m: {}, c: {}, learning rate: {}, iteration: {}, cost: {}".format(
        best_m, best_c, best_lr, best_iteration, best_cost))

  
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
grad_des(x,y)

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  
y = np.array([5, 7, 9, 11, 13])


model = LinearRegression()


model.fit(x, y)


m = model.coef_[0]
c = model.intercept_

print("Slope (m):", m)
print("Intercept (c):", c)


