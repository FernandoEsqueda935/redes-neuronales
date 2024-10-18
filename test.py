import numpy as np
import functions as fn
from sklearn.model_selection import train_test_split

#crea una matriz de 3 filas y 3 columna, las primeras 2 filas son los pesos de la primera capa junto con su bias y la tercera es es para la ultima capa
##wh = np.random.rand(5, 3)

wh = np.ones((5, 3))


inputs = np.loadtxt("linreg_dataset.dat", usecols=(0, 1))  # Lee las columnas 0 y 1
targets = np.loadtxt("linreg_dataset.dat", usecols=2)  # Lee la columna 2

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.05, random_state=42)

datos = inputs_train.shape[0]

epocas = 1000

lr = 0.001

'''
a0 = np.vstack(inputs_train[0])
a0 = np.vstack((a0, np.ones(a0.shape[1])))

n1 = np.dot(wh[:2], a0)

a1 = fn.heaviside(n1)
a1 = np.vstack((a1, np.ones(a0.shape[1])))

n2 = np.dot(wh[2:3], a1)

a2 = fn.heaviside(n2) 
print(a2)
'''

funciones_activacion = [fn.lineal , fn.lineal]

funciones_activacion_d = [fn.lineal_derivative , fn.lineal_derivative]

print("Pesos iniciales: ", wh.shape)
print("Pesos primera capa", wh[:2].shape)
print("Pesos segunda capa", wh[2:3].shape)

for epoca in range(epocas):
    for dato in range(datos):
        
        a0 = np.vstack([[.5] , [.5]])
        ah0 = np.vstack((a0, np.ones(a0.shape[1])))
        
        print("wh1: \n" , wh[:2])
        print("ah0: \n", ah0)

        n1 = wh[:2] @ ah0

        a1 = funciones_activacion[0](n1)

        print("a1: \n", a1)

        ah1 = np.vstack((a1, np.ones(a1.shape[1])))
        
        print("wh2: \n" , wh[2:4])
        print("ah1: \n", ah1)

        n2 = wh[2:4 , :] @ ah1

        a2 = funciones_activacion[0](n2)

        print("a2: \n", a2)

        ah2 = np.vstack((a2, np.ones(a2.shape[1])))
        
        print("wh3: \n" , wh[4:5])
        print("ah2: \n", ah2)

        n3 = wh[4:5 , :] @ ah2

        a3 = funciones_activacion[1](n3) 

        print("a3: \n", a3)

        error = 8 - a3

        print("Error: \n", error)
        
        SSE = error.T @ error 
        
        print("SSE: \n", SSE)
        
        delta3 = ([-2] @ error) * funciones_activacion_d[1](n3)
        
        print("delta3: \n", delta3)
        
        delta3 = delta3.reshape(1, 1)
        
        g3 = delta3 @ ah2.T
        
        print("g3: \n", g3)
        
        delta2 = funciones_activacion_d[0](n2) * (wh[4:5 , :2].T @ delta3)
        
        print("delta2: \n", delta2)
        
        g2 = delta2 @ ah1.T
        
        print("g2: \n", g2)
                
        delta1 = funciones_activacion_d[0](n1) * (wh[2:4 , :2].T @ delta2)
        
        print("delta1: \n", delta1)
        
        g1 = delta1 @ ah0.T
        
        print("g1: \n", g1)
        
        wh[:2 , :] = wh[:2 , :] - lr * g1
        
        print("wh1: \n" , wh[:2])
        
        wh[2:4 , :] = wh[2:4 , :] -lr * g2
        
        print("wh2: \n" , wh[2:4])
        
        wh[4:5 , :] = wh[4:5 , :] - lr * g3
        
        print("wh3: \n" , wh[4:5])
        
        
        print("SSE: ", SSE)
        if (dato == 100):
            break
    break
    if (epoca%100 == 0):
        print("wh: " , wh)

'''        n1 = np.dot(wh[:2], a0)

        a1 = funciones_activacion[0](n1)
        a1 = np.vstack((a1, np.ones(a0.shape[1])))

        n2 = np.dot(wh[2:3], a1)

        a2 = funciones_activacion[1](n2)

        error = targets_train[dato] - a2
        
        delta2 = (-2 * error) @ funciones_activacion_d[1](n2)
        
        g2 = delta2 * a1.T

        wh[2:3] += - lr * g2
        
        delta1 =(delta2 * wh[2:3]) @ funciones_activacion_d[0](n2)
        
        g1 = delta1 * a0.T

        wh[:2] += -lr * g1'''
'''
print("SSE: ", SSE)
      
for dato in range(targets_test.shape[0]):
    a0 = np.vstack(inputs_train[dato])
    ah0 = np.vstack((a0, np.ones(a0.shape[1])))
    
    n1 = wh[:2] @ ah0

    a1 = funciones_activacion[0](n1)

    ah1 = np.vstack((a1, np.ones(a1.shape[1])))

    n2 = wh[2:4 , :] @ ah1

    a2 = funciones_activacion[0](n2)
    ah2 = np.vstack((a2, np.ones(a2.shape[1])))
    
    n3 = wh[4:5 , :] @ ah2

    a3 = funciones_activacion[1](n3) 
    
    print(dato, "." , "Salida:", a3 , "Target:", targets_test[dato] , "Error:", (targets_test[dato] - a3))'''