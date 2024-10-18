import numpy as np

class neural_network:
    def __init__(self, layers_density, q, activation_functions, lr, optimizer):
        wh , ah = self.get_arch(layers_density, q)
        self.n = []
        self.gradientes = []
        self.arch = network_architecture(wh, ah)
        self.layers = len(layers_density)
        self.deltas = np.zeros(self.layers - 1)
        self.activation_functions = activation_functions
        self.lr = lr
        self.optimizer = optimizer
        self.output = np.zeros((q, layers_density[self.layers-1]))
        
    def get_arch(self, layers_density, q):
        wh = []
        ah = []
        next_layer = 0
        for neurons in layers_density:
            next_layer += 1
            if (next_layer < len(layers_density)):
                wh_temp = np.random.rand(layers_density[next_layer] , neurons + 1)
                ah_temp = np.ones((neurons, q))
                wh.append(wh_temp)
                ah.append(ah_temp)
        return wh , ah    
    
    def foward_propagation(self, inputs):
        self.arch.ah[0] = inputs 
            
        for layer in range(self.layers-1):
            self.arch.ah[layer] = np.append(self.arch.ah[layer], np.ones((1, inputs.shape[1])), axis=0)    
            ##print("wh: \n",self.arch.wh[layer])
            ##print("ah: \n",self.arch.ah[layer])
            n_temp = self.arch.wh[layer] @ self.arch.ah[layer]
            ah_temp = n_temp
            ##print("ah_temp: \n",ah_temp)
            self.n.append(n_temp)
            if layer < self.layers-2:
                self.arch.ah[layer+1] = self.activation_functions[layer](ah_temp)
            else:
                self.output = ah_temp
    def back_propagation(self, targets):
        e = targets - self.output
        print("e: ",e)
        SEE = e.T @ e
        print("SEE: ", SEE)
        max_index  = self.layers - 2
        for index in range(max_index + 1):
            if index == 0:
                self.deltas[max_index - index] = -2 * (e * self.activation_functions[max_index - index](self.n[max_index - index]))
            else: 
                self.deltas[max_index - index] = (self.arch.wh[max_index - index + 1].T @ self.deltas[max_index - index + 1]) * self.activation_functions[max_index - index](self.n[max_index - index])
            
            
            print("deltas: ", self.deltas[max_index - index])
            print("ah: ", self.arch.ah[max_index - index])
            
            gradiente = self.deltas[max_index - index] * self.arch.ah[max_index - index][:-1, :].T
            self.gradientes.append(gradiente)
            
class network_architecture: 
    
    def __init__ (self, wh, ah):
        self.wh = wh
        self.ah = ah
        