import numpy as np
from dataclasses import dataclass

@dataclass
class Activations:
    def leaky_relu(self, input: np.asarray, alpha=0.01) -> np.asarray:
        return np.where(input > 0, input, alpha * input)
    
    def leaky_relu_der(self, input: np.asarray, alpha=0.01) -> np.asarray:
        return np.where(input > 0, 1, alpha)

    def reLU(self, input):
        return np.maximum(0, input)
    def reLU_der(self, input):
        return (input > 0).astype(float)
    
    def softmax(self, input: np.asarray) -> np.asarray:
        z_max = np.maximum(input, axis=-1, keepdims=True) 

        exp = np.exp(input - z_max)
        exp_sum = np.sum(exp, axis=-1, keepdims=True)

        return exp/exp_sum
    
    def sigmoid(self, input: np.asarray) -> np.asarray:
        exp = np.exp(-input)
        return 1/(1+exp)
    
    def sigmoid_der(self, input: np.asarray) -> np.asarray:
        exp = np.exp(-input)
        sigmoid = 1/(1+exp)

        return sigmoid * (1-sigmoid)

    def stable_sigmoid(self, input:np.asarray) -> np.asarray:
        return np.where(input >= 0, 1/(1+np.exp(-input)), np.exp(input)/(np.exp(input)+1))
    
    def stable_sigmoid_der(self, input: np.asarray) -> np.asarray:
        st_sig = np.where(input >= 0, 1/(1+np.exp(-input)), np.exp(input)/(np.exp(input)+1))
        return st_sig * (1 - st_sig)
    
    