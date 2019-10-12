import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

relu_v = np.vectorize(relu)

def leaky_relu(x, a):
    return np.maximum(x, a*x)

leaky_v = np.vectorize(leaky_relu)

def elu(x, a):
    if x >= 0:
        return x
    else:
        return a*(np.exp(x) - 1)

elu_v = np.vectorize(elu)

x = np.linspace(-5, 5, 1000)

plt.grid()
plt.plot(x, sigmoid(x))
plt.show()
plt.grid()
plt.plot(x, tanh(x))
plt.show()
plt.grid()
plt.plot(x, relu_v(x))
plt.show()
plt.grid()
plt.plot(x, leaky_v(x, 0.1))
plt.show()
plt.grid()
plt.plot(x, elu_v(x, 0.1))
plt.show()