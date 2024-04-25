import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


min_error_diff = 1e-8

x1 = 0.04
x2 = 0.2

y_true = 0.5

lr = 0.4

np.random.seed(23)
w1, w2, w3, w4, w5, w6 = np.random.rand(6).tolist()
print("Initial weights:")
print("w1: ", w1)
print("w2: ", w2)
print("w3: ", w3)
print("w4: ", w4)
print("w5: ", w5)
print("w6: ", w6)
print("")

error = None

i = 1
while True:
    # forward pass
    sum_h1 = w1 * x1 + w2 * x2
    out_h1 = sigmoid(sum_h1)
    sum_h2 = w3 * x1 + w4 * x2
    out_h2 = sigmoid(sum_h2)
    y_pred = w5 * out_h1 + w6 * out_h2
    out_y_pred = sigmoid(y_pred)

    # calculate error
    last_error = error
    error = 0.5 * (y_true - out_y_pred) ** 2

    print(f"Iteration: {i}, Error: {error}")

    if last_error is not None and abs(last_error - error) < min_error_diff:
        break

    # backward pass

    # calculate direct connection gradients
    d_err_d_out_y_pred = (-1) * (y_true - out_y_pred)

    d_out_y_pred_d_y_pred = out_y_pred * (1 - out_y_pred)

    d_y_pred_d_w5 = out_h1
    d_y_pred_d_w6 = out_h2

    d_y_pred_d_out_h1 = w5
    d_y_pred_d_out_h2 = w6

    d_out_h1_d_h1 = out_h1 * (1 - out_h1)
    d_out_h2_d_h2 = out_h2 * (1 - out_h2)

    d_h1_d_w1 = x1
    d_h1_d_w2 = x2

    d_h2_d_w3 = x1
    d_h2_d_w4 = x2

    # calculate weight gradients

    d_err_d_w1 = (
        d_err_d_out_y_pred
        * d_out_y_pred_d_y_pred
        * d_y_pred_d_out_h1
        * d_out_h1_d_h1
        * d_h1_d_w1
    )
    d_err_d_w2 = (
        d_err_d_out_y_pred
        * d_out_y_pred_d_y_pred
        * d_y_pred_d_out_h1
        * d_out_h1_d_h1
        * d_h1_d_w2
    )
    d_err_d_w3 = (
        d_err_d_out_y_pred
        * d_out_y_pred_d_y_pred
        * d_y_pred_d_out_h2
        * d_out_h2_d_h2
        * d_h2_d_w3
    )
    d_err_d_w4 = (
        d_err_d_out_y_pred
        * d_out_y_pred_d_y_pred
        * d_y_pred_d_out_h2
        * d_out_h2_d_h2
        * d_h2_d_w4
    )
    d_err_d_w5 = d_err_d_out_y_pred * d_out_y_pred_d_y_pred * d_y_pred_d_w5
    d_err_d_w6 = d_err_d_out_y_pred * d_out_y_pred_d_y_pred * d_y_pred_d_w6

    # gradient descent step
    w1 -= lr * d_err_d_w1
    w2 -= lr * d_err_d_w2
    w3 -= lr * d_err_d_w3
    w4 -= lr * d_err_d_w4
    w5 -= lr * d_err_d_w5
    w6 -= lr * d_err_d_w6

    i += 1

"""
Iteration: 1, Error: 0.006935990239085405
Iteration: 2, Error: 0.006757994362320645
Iteration: 3, Error: 0.006584065883545012
Iteration: 4, Error: 0.006414136694760013
Iteration: 5, Error: 0.00624813859377414
Iteration: 6, Error: 0.006086003356807131
Iteration: 7, Error: 0.005927662807821905
Iteration: 8, Error: 0.005773048884604081
Iteration: 9, Error: 0.005622093701617335
Iteration: 10, Error: 0.0054747296096708436

.
.
.

Iteration: 347, Error: 3.365267594710107e-07
"""
