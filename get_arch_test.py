import get_arch as ga
import numpy as np
import functions as f

red_neuronal = ga.neural_network([2, 3, 1, 3, 4, 1], 1, [f.lineal, f.lineal, f.lineal, f.lineal, f.lineal], 0.1, 'sgd')
red_neuronal.foward_propagation(np.array([[3], [.5]]))
red_neuronal.back_propagation(np.array([[1]]))

print (red_neuronal.arch.wh)
print (red_neuronal.arch.ah)


##print(red_neuronal.gradientes)

'''print(red_neuronal.output)
'''

'''index = len(red_neuronal.gradientes) - 1
for i in range(len(red_neuronal.gradientes)):
    print("gradiente", i, red_neuronal.gradientes[i].shape)
    print("pesos", red_neuronal.arch.wh[index - i].shape)'''
    

    
