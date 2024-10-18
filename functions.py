import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def gaussian(x):
    return np.exp(-x**2 / 2)

def gaussian_derivative(x):
    return -x * gaussian(x)

def heaviside(x):
    return np.heaviside(x, 1)

def heaviside_derivative(x):
    return 0

def lineal(x):
    return x

def lineal_derivative(x):
    return 1

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2