from helper import *
import numpy as np
import argparse

n_out = 10
n_hidden = 16
hid_layers = 5
ix = "relu"
init = "Xavier"


activation = {"tanh": Tanh(), "relu": Relu(), "sigmoid": Sigmoid()}

model = Sequential([Linear(784, n_hidden, weight_init = init)])
model.append(activation[ix])

model.append(Linear(n_hidden, 10, weight_init = init))

x = np.random.randn(32, 28, 28)
x = x.reshape(32, -1)
y = np.random.randint(0, 9, (32,))
y = np.eye(10)[y]
opt = Optimizer(lr=1e-3, optimizer="rmsprop")

#Forward Pass. ## Our cross entropy takes care of softmax activation in final layer

logits = model(x)
logits = logits/np.sum(logits, axis=-1, keepdims=True) # To stabilize the backward pass
loss = CrossEntropyLoss()(logits, y)
# print(loss)
#Backward Pass

dout = CrossEntropyLoss().grad(logits, y)
dout = model.backward(dout)

#parameter update
opt(model.parameters(), dout[1])

print("Loss: ", loss)

