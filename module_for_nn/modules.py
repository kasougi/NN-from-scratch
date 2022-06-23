from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class myModule(object):
    def __init__(self):
        self.output = None
        self.training = None
        self.gradient = None
        
    def forward(self, inpt):
        return self.update_output(inpt)
    
    def backward(self,inpt, grad_output):
        self.update_grad_inpt(inpt, grad_output)
        self.acc_grad_parameters(inpt, grad_output)
        return self.grad_inpt
    
    def zero_grad_parameters(self):
        pass
    
    def get_parameters(self):
        return []
    
    def get_grad_parameters(self):
        return []
    
    def train(self):
        pass
    
    def acc_grad_parameters(self, inpt, gradOutput):
        pass
    
    def evaluate(self):
        pass
    
    
    def __repr__(self):
        return "Module"


class Sequential(myModule):
    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        
    def add(self, module):
        self.modules.append(module)
        
    def update_output(self, inpt):
        self.output = inpt
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, inpt, grad_output):
        for i in range(len(self.modules)-1, 0, -1):
            grad_output = self.modules[i].backward(self.modules[i-1].output, grad_output)
        self.grad_inpt = self.modules[0].backward(inpt, grad_output)
        return self.grad_inpt
    
    def zero_grad_parameters(self):
        for module in self.modules:
            module.zero_grad_parameters()
    
    def get_parameters(self):
        return [i.get_parameters() for i in self.modules]
    
    def get_grad_parameters(self):
        return [i.get_grad_parameters() for i in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + "\n" for x in self.modules])
        return string
        
    def get_models(self):
        return self.modules

    def __getitem__(self, x):
        return self.modules.__getitem__(x)
    
    def train(self):
        self.training = True
        for module in self.modules:
            module.train()
            
    def evaulate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()





class Linear(myModule):
    
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv,stdv, size = (n_out, n_in))
        self.B = np.random.uniform(-stdv,stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradB = np.zeros_like(self.B)
        
    def update_output(self, inpt):
    
        self.output = inpt@self.W.T + self.B
        return self.output
    
    def update_grad_inpt(self, inpt, grad_output):
        self.grad_inpt = grad_output @ self.W
        
    def acc_grad_parameters(self, inpt, grad_output):
        self.gradW = np.sum(inpt[:, None, :] * grad_output[:, :, None], axis=0) / inpt.shape[0]
        self.gradB = np.sum(grad_output, axis=0) / grad_output.shape[0]
    
    def zero_grad_parameters(self):
        self.gradW.fill(0)
        self.gradB.fill(0)
        
    def get_parameters(self):
        return [self.W, self.B]
    
    def get_grad_parameters(self):
        return [self.gradW, self.gradB]
    
    def __repr__(self):
        s = self.W.shape
        q = f'Linear {s[1], s[0]}'
        return q

class Criterion(object):
    def __init__ (self):
        self.output = None
        self.grad_inpt = None

    def forward(self, inpt, target):
        return self.update_output(inpt, target)

    def backward(self, inpt, target):
        return self.update_grad_inpt(inpt, target)

    def update_output(self, inpt, target):
        return self.output

    def update_grad_inpt(self, inpt, target):
        return self.grad_inpt

    def __repr__(self):
        return "Criterion"


class ClassNLLCriterion(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def update_output(self, inpt, target): 
        global AAA
        AAA = inpt
        inpt = np.clip(inpt, self.EPS, 1 - self.EPS)
        grid = inpt[target > 0]
        l_out = -np.log(grid)
        self.output = np.sum(l_out)/ inpt.shape[0]
        return self.output

    def update_grad_inpt(self, inpt, target):
        inpt = np.clip(inpt, self.EPS, 1 - self.EPS)
            
        new_inpt = 1 / inpt
        target = 1*target
        self.grad_inpt = -1*target * new_inpt
        return self.grad_inpt
    
    def __repr__(self):
        return "ClassNLLCriterion"


class SoftMax(myModule):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def sftmx(self, inpt):
        new_inpt = np.subtract(inpt, inpt.max(axis=1, keepdims=True))
        
        new_inpt = np.exp(new_inpt)
        ans = new_inpt / np.sum(new_inpt, axis=1, keepdims=True)
        return ans
    
    def update_output(self, inpt):
        return self.sftmx(inpt)
    
    
    def update_grad_inpt(self, inpt, grad_output):
        
        p_inpt = self.sftmx(inpt)
        tg_s1 = p_inpt[:,np.newaxis,:] * np.diag(np.ones(p_inpt.shape[1]))[np.newaxis, :]
        tg_s2 = p_inpt[:,:,np.newaxis] @ p_inpt[:, np.newaxis, :]
        final_s = tg_s1 - tg_s2
        self.grad_inpt =  np.squeeze((grad_output[:, np.newaxis, :] @ final_s), axis=1)
        return self.grad_inpt


    def __repr__(self):
        return "SoftMax"


class ReLU(myModule):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def update_output(self, inpt):
        self.output = np.maximum(inpt, 0)
        return self.output
    
    def update_grad_inpt(self, inpt, gradOutput):
        self.grad_inpt = np.multiply(gradOutput , inpt > 0)
        return self.grad_inpt
    
    def __repr__(self):
        return "ReLU"

class LeakyReLU(myModule):
    def __init__(self, a = 0.3):
        super(LeakyReLU, self).__init__()
        self.a = a
        
    def update_output(self, inpt):
        self.gradInput = np.maximum(inpt, inpt*self.a)
        return self.output
    
    def update_grad_inpt(self, inpt, grad_output):
        self.gradInput = np.multiply(grad_output , inpt > 0) + np.multiply(grad_output * self.a, inpt <= 0 )
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"



