import numpy as np
import functions as fn
from sklearn.model_selection import train_test_split

# Crea una matriz de 2 filas y 1 columna con valores aleatorios
w = np.random.rand(2, 1)
# Pone abajo de la matriz ya existente una fila de unos de la cantidad de columnas de W
wh = np.vstack((w, np.ones(w.shape[1])))

inputs = np.loadtxt("xor_dataset.dat", usecols=(0, 1))  # Lee las columnas 0 y 1
targets = np.loadtxt("xor_dataset.dat", usecols=2)  # Lee la columna 2

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1, random_state=42)

datos = inputs_train.shape[0]
epocas = 1000

lr = 0.001
for epoca in range(epocas):
    for dato in range(datos):
        a0 = np.vstack(inputs_train[dato])
        a0 = np.vstack((a0, np.ones(a0.shape[1])))

        n1 = np.dot(wh.T, a0)
        a1 = fn.sigmoid(n1)

        error = targets_train[dato] - a1
        delta = -2 * error * fn.sigmoid_derivative(n1)
        g = delta * a0
        wh += - lr * g
        
epocas = inputs_test.shape[0]

for epoca in range(epocas):
    a0 = np.vstack(inputs_test[epoca])
    a0 = np.vstack((a0, np.ones(a0.shape[1])))

    n1 = np.dot(wh.T, a0)
    a1 = fn.sigmoid(n1)
    print("Salida:", a1 , "Target:", targets_test[epoca] , "Error:", (targets_test[epoca] - a1))


