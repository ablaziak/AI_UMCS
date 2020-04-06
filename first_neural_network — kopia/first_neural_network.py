import random
from math import exp

class Neuron:
    w0 = random.random()
    w1 = random.random()
    w2 = random.random()
    w3 = random.random()
    w4 = random.random()
    w5 = random.random()

    b0 = random.random()
    b1 = random.random()
    b2 = random.random()

    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def derivsigmoid(self, x):
        return self.sigmoid(self, x)*(1-self.sigmoid(self, x))

    def mse_loss(self, y, ypred):
        return ((y-ypred)**2)/2

    def derivmse_loss(self, y, ypred):
        return -2*(y-ypred)

    def feedforward(self, x):
        h0 = self.sigmoid(self, self.w0*x[0]+self.w1*x[1]+self.b0)
        h1 = self.sigmoid(self, self.w2 * x[0] + self.w3 * x[1] + self.b1)
        o0 = self.sigmoid(self, self.w4 * h0 + self.w5 * h1 + self.b2)
        return o0

    def train(self, data, ypred, q, epochs):
        result = [0, 0, 0, 0]
        for j in range(0, epochs):
            for i in range (0,4):
                result[i] = self.feedforward(self, data[i])
                if j%10 == 0:
                    print(i, ": ", result[i])
                mse = self.mse_loss(self, result[i], ypred[i])
                deriv_mse = self.derivmse_loss(self, result[i], ypred[i])

                dw5 = (self.w2 * data[i][0] + self.w3 * data[i][1] + self.b1) * self.derivsigmoid(self, result[i])
                dw4 = (self.w0 * data[i][0] + self.w1 * data[i][1] + self.b0) * self.derivsigmoid(self, result[i])
                db2 = self.derivsigmoid(self, result[i])
                dh0 = dw4 * self.derivsigmoid(self, result[i])
                dh1 = dw5 * self.derivsigmoid(self, result[i])
                dw0 = self.derivsigmoid(self, self.w0 * data[i][0] + self.w1 * data[i][1] + self.b0) * data[i][0]
                dw1 = self.derivsigmoid(self, self.w0 * data[i][0] + self.w1 * data[i][1] + self.b0) * data[i][1]
                dw2 = self.derivsigmoid(self, self.w2 * data[i][0] + self.w3 * data[i][1] + self.b1) * data[i][0]
                dw3 = self.derivsigmoid(self, self.w2 * data[i][0] + self.w3 * data[i][1] + self.b1) * data[i][1]
                db0 = self.derivsigmoid(self, self.w0 * data[i][0] + self.w1 * data[i][1] + self.b0)
                db1 = self.derivsigmoid(self, self.w2 * data[i][0] + self.w3 * data[i][1] + self.b1)

                self.w0 -= q*dw0*dh0*deriv_mse
                self.w1 -= q*dw1*dh0*deriv_mse
                self.w2 -= q*dw2*dh1*deriv_mse
                self.w3 -= q*dw3*dh1*deriv_mse
                self.b0 -= q*db0*dh0*deriv_mse
                self.b1 -= q*db1*dh1*deriv_mse
                self.w4 -= q*dw4*deriv_mse
                self.w5 -= q*dw5*deriv_mse
                self.b2 -= q*db2*deriv_mse

                ##if j%10 == 0:
                    ##print(i," : ", result[i])



learn_rate = 0.1
epochs = 1000
data = [[0,0], [0,1], [1,0], [1,1]]
ypred = [0,0,0,1]

Neuron.train(Neuron, data, ypred, learn_rate, epochs)