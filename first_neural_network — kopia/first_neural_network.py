##import random
from math import exp
import numpy as np


class Neuron:
    w0 = np.random.normal()
    w1 = np.random.normal()
    w2 = np.random.normal()
    w3 = np.random.normal()
    w4 = np.random.normal()
    w5 = np.random.normal()

    b0 = np.random.normal()
    b1 = np.random.normal()
    b2 = np.random.normal()

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def derivsigmoid(self, x):
        return self.sigmoid(self, x) * (1 - self.sigmoid(self, x))

    def mse(self, target_value, y):
        return ((target_value - y) ** 2) / 2

    def derivmse(self, target_value, y):
        return -2 * (target_value - y)

    def feedforward(self, x):
        h0 = self.sigmoid(self, self.w0 * x[0] + self.w1 * x[1] + self.b0)
        h1 = self.sigmoid(self, self.w2 * x[0] + self.w3 * x[1] + self.b1)
        o0 = self.sigmoid(self, self.w4 * h0 + self.w5 * h1 + self.b2)
        return o0

    def printall(self, data, result, loss):
        for i in range (0,4):
            print("[", data[i][0], ", ", data[i][1], "]", end=' | ')
        print()
        for i in range(0,4):
            print("%.8f" %result[i], end=' | ')
        print()
        for i in range(0,4):
            print("loss: %.3f" %loss[i], end=' |')
        print("\n")


    def train(self, data, target_value, learn_rate, epochs):
        result = [0, 0, 0, 0]
        loss = [0, 0, 0, 0]
        for j in range(0, epochs):
            for i in range(0, 4):
                sum_h0 = self.w0 * data[i][0] + self.w1 * data[i][1] + self.b0
                sum_h1 = self.w2 * data[i][0] + self.w3 * data[i][1] + self.b1

                h0 = self.sigmoid(self, sum_h0)
                h1 = self.sigmoid(self, sum_h1)

                sum_o0 = self.w4 * h0 + self.w5 * h1 + self.b2
                o0 = self.sigmoid(self, sum_o0)

                ## mse = self.mse_loss(self, result[i], ypred[i])

                deriv_mse = self.derivmse(self, target_value[i], o0)

                dw5 = h1 * self.derivsigmoid(self, sum_o0)
                dw4 = h0 * self.derivsigmoid(self, sum_o0)
                db2 = self.derivsigmoid(self, sum_o0)

                dh0 = self.w4 * self.derivsigmoid(self, sum_o0)
                dh1 = self.w5 * self.derivsigmoid(self, sum_o0)

                dw0 = self.derivsigmoid(self, sum_h0) * data[i][0]
                dw1 = self.derivsigmoid(self, sum_h0) * data[i][1]
                db0 = self.derivsigmoid(self, sum_h0)

                dw2 = self.derivsigmoid(self, sum_h1) * data[i][0]
                dw3 = self.derivsigmoid(self, sum_h1) * data[i][1]
                db1 = self.derivsigmoid(self, sum_h1)

                self.w0 -= learn_rate * dw0 * dh0 * deriv_mse
                self.w1 -= learn_rate * dw1 * dh0 * deriv_mse
                self.b0 -= learn_rate * db0 * dh0 * deriv_mse

                self.w2 -= learn_rate * dw2 * dh1 * deriv_mse
                self.w3 -= learn_rate * dw3 * dh1 * deriv_mse
                self.b1 -= learn_rate * db1 * dh1 * deriv_mse

                self.w4 -= learn_rate * dw4 * deriv_mse
                self.w5 -= learn_rate * dw5 * deriv_mse
                self.b2 -= learn_rate * db2 * deriv_mse
                if j%10 == 0:
                    result[i] = self.feedforward(self, data[i])
                    loss[i] = self.mse(self, target_value[i], result[i])
            if  j%10==0:
                print("epoch: ", j)
                self.printall(self, data, result, loss)


learn_rate = 0.1
epochs = 1000
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_value = [0, 1, 1, 1]

Neuron.train(Neuron, data, target_value, learn_rate, epochs)
