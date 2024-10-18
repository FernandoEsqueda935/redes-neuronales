import numpy as np
import functions as fn
from sklearn.model_selection import train_test_split

wh = np.random.rand(5, 3)

inputs = np.loadtxt("linreg_dataset.dat", usecols=(0, 1)) 
targets = np.loadtxt("linreg_dataset.dat", usecols=2)  

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3, random_state=42)
datos = inputs_train.shape[0]

min_train = np.min(inputs_train[:, 0:1], axis=0)
max_train = np.max(inputs_train[:, 0:1], axis=0)
inputs_train[:, 0:1] = 2 * ((inputs_train[:, 0:1] - min_train) / (max_train - min_train)) - 1

min_train = np.min(inputs_train[:, 1:2], axis=0)
max_train = np.max(inputs_train[:, 1:2], axis=0)
inputs_train[:, 1:2] = 2 * ((inputs_train[:, 1:2] - min_train) / (max_train - min_train)) - 1

min_test = np.min(inputs_train[:, 0:1], axis=0)
max_test = np.max(inputs_train[:, 0:1], axis=0)

inputs_test [: , 0:1] = 2 * ((inputs_test[:, 0:1] - min_test) / (max_test - min_test)) - 1

min_test = np.min(inputs_train[:, 1:2], axis=0)
max_test = np.max(inputs_train[:, 1:2], axis=0)
inputs_test [: , 1:2] = 2 * ((inputs_test[:, 1:2] - min_test) / (max_test - min_test)) - 1

lr = 0.001
b1 = 0.999
b2 = 0.99999
epsilon = 1e-8
wd = 0

m = 0
v = 0
vh = 0


epocas = 10000

funciones_activacion = [fn.swish, fn.lineal]
funciones_activacion_d = [fn.swish_derivative, fn.lineal_derivative]

for epoca in range(epocas):
    for dato in range(datos):
        
        a0 = np.vstack(inputs_train[dato])
        ah0 = np.vstack((a0, np.ones(a0.shape[1])))
        n1 = wh[:2] @ ah0
        a1 = funciones_activacion[0](n1)
        ah1 = np.vstack((a1, np.ones(a1.shape[1])))
        n2 = wh[2:4, :] @ ah1
        a2 = funciones_activacion[0](n2)
        ah2 = np.vstack((a2, np.ones(a2.shape[1])))
        n3 = wh[4:5, :] @ ah2
        a3 = funciones_activacion[1](n3)
        

        error = targets_train[dato] - a3
        SSE = error.T @ error 
        
        delta3 = (-2 * error) * funciones_activacion_d[1](n3)
        delta3 = delta3.reshape(1, 1)
        g3 = delta3 @ ah2.T
        
        delta2 = funciones_activacion_d[0](n2) * (wh[4:5, :2].T @ delta3)
        g2 = delta2 @ ah1.T
        
        delta1 = funciones_activacion_d[0](n1) * (wh[2:4, :2].T @ delta2)
        g1 = delta1 @ ah0.T

        wh_v = np.concatenate([w.ravel() for w in wh])
        
        g_v = np.concatenate([g.ravel() for g in [g1, g2, g3]])
        
        if wd != 0:
            g_v = g_v + wd * wh_v
            
        m = b1 * m + (1 - b1) * g_v
        v = b2 * v + (1 - b2) * (g_v ** 2)
        
        mh = m/(1 - (b1 ** (epoca + 1)))
        vh = v/(1 - (b2 ** (epoca + 1)))
        
        wh_v -= lr * mh / (np.sqrt(vh) + epsilon)
        
        wh = wh_v.reshape(wh.shape)
        '''
        for t, (g, m, v, wh_slice) in enumerate(zip([g1, g2, g3], [m1, m2, m3], [v1, v2, v3], [wh[:2], wh[2:4], wh[4:5]])):
            m[:] = beta1 * m + (1 - beta1) * g  
            v[:] = beta2 * v + (1 - beta2) * (g ** 2)  
            
            m_hat = m / (1 - beta1**(epoca + 1)) 
            v_hat = v / (1 - beta2**(epoca + 1))  
            
            wh_slice -= lr * m_hat / (np.sqrt(v_hat) + epsilon) '''

    if epoca % 100 == 0:
        print(f"Época {epoca} - SSE: {SSE}")
'''
for dato in range(targets_test.shape[0]):
    a0 = np.vstack(inputs_train[dato])
    ah0 = np.vstack((a0, np.ones(a0.shape[1])))
    n1 = wh[:2] @ ah0
    a1 = funciones_activacion[0](n1)
    ah1 = np.vstack((a1, np.ones(a1.shape[1])))
    n2 = wh[2:4, :] @ ah1
    a2 = funciones_activacion[0](n2)
    ah2 = np.vstack((a2, np.ones(a2.shape[1])))
    n3 = wh[4:5, :] @ ah2
    a3 = funciones_activacion[1](n3)
    
    print(f"Dato {dato} - Salida: {a3} - Target: {targets_test[dato]} - Error: {targets_test[dato] - a3}")'''
