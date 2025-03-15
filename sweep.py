'''
Aayushman | DA24S016
DA6401: Assignment 1
'''

from helper import *
import numpy as np
import wandb
import argparse
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load the Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1)


def initialize_model(activation, layer_sizes, init="Xavier"):
    if activation == "tanh":
      print("Activation used is Tanh")

      model = Sequential([Linear(784, layer_sizes[1], weight_init=init)])

      # Hidden layers
      for i in range(1, len(layer_sizes)-1):
          model.append(Tanh())
          model.append(Linear(layer_sizes[i], layer_sizes[i+1], weight_init=init))

      # Final output layer
      model.append(Tanh())
      model.append(Linear(layer_sizes[-1], 10, weight_init=init))  # 128->10


    elif activation == "relu":
      print("Activation used is ReLu")

      model = Sequential([Linear(784, layer_sizes[1], weight_init=init)])

      # Hidden layers
      for i in range(1, len(layer_sizes)-1):
          model.append(Relu())
          model.append(Linear(layer_sizes[i], layer_sizes[i+1], weight_init=init))

      # Final output layer
      model.append(Relu())
      model.append(Linear(layer_sizes[-1], 10, weight_init=init))  # 128->10


    elif activation == "sigmoid":
      print("Activation used is Sigmoid")

      model = Sequential([Linear(784, layer_sizes[1], weight_init=init)])

      # Hidden layers
      for i in range(1, len(layer_sizes)-1):
          model.append(Sigmoid())
          model.append(Linear(layer_sizes[i], layer_sizes[i+1], weight_init=init))

      # Final output layer
      model.append(Sigmoid())
      model.append(Linear(layer_sizes[-1], 10, weight_init=init))  # 128->10
    return model

 # -----------------------------------------------------------------------------------------------


def train():
    #init run
    run = wandb.init()

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1)
    X = train_images.reshape(train_images.shape[0], -1)/ 255.0
    Y = train_labels
    Y = np.eye(10)[Y]


    config = wandb.config
    hid_layers = config.hid_layers
    init = config.init
    # max_steps = wandb.config.n_steps
    batch_size = config.batch_size
    activation = config.activation
    nepoch = config.nepoch
    loss_fn = config.loss
    Loss = CrossEntropyLoss() if loss_fn=="cross_entropy" else MSE()
    #naming the run
    run.name = f"opt_{config.optimizer}|loss_{loss_fn}|lr={config.lr}|batch_{batch_size}|act_{activation}|hid_{hid_layers}|neurons_{config.hid_size}|nrns_{nepoch}|init_{init}" + str(np.random.randint(1000))

    layer_sizes = [config.hid_size] * (hid_layers + 1)

    

    model = initialize_model(activation, layer_sizes, init)

    opt = Optimizer(lr=config.lr, optimizer=config.optimizer, decay = config.decay, param=model.parameters())

    logits = model(X)
    train_loss = Loss(logits, Y)
    train_accuracy = np.mean(np.argmax(logits, axis=1) == np.argmax(Y, axis=1))

    val_logits = model(val_images.reshape(val_images.shape[0], -1)/ 255.0)
    Yv = np.eye(10)[val_labels]
    val_loss = Loss(val_logits, Yv)
    val_accuracy = np.mean(np.argmax(val_logits, axis=1) == np.argmax(Yv, axis=1))

    print(f"Start of Training: {1} Train Accuracy: {train_accuracy:.4f} Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.4f}")
    wandb.log({"Epoch": 0, "Val Loss": val_loss, "Train Accuracy": train_accuracy, "Val Accuracy": val_accuracy})

    for epoch in range(nepoch):
      print("-------x-------")
      #Shuffling
      indices = np.random.permutation(X.shape[0])
      X = X[indices]
      Y = Y[indices]

      for i in range(0, train_images.shape[0], batch_size):
        Xb = X[i:i + batch_size]
        Yb = Y[i:i + batch_size]

        logits = model(Xb)

        loss = Loss(logits, Yb)

        #Backward Pass
        dout = Loss.grad(logits, Yb)
        dout = model.backward(dout)


        batch_num = i//batch_size
        total_batch = train_images.shape[0]//batch_size



        #Parameter Update
        opt(model.parameters(), dout[1])

        if batch_num%200 == 0: # print every once in a while uhh to be precise after 200 batch
          print(f'Epoch({epoch+1}/{nepoch})\t Batch({batch_num:2d}/{total_batch:2d}): \tTrain Loss  {loss:.4f}')

      opt.t += 1


      #Accuracy Calculation
      wandb.log({"Epoch": epoch+1, "Train Loss": loss})
      logits = model(X)
      train_loss = Loss(logits, Y)
      train_accuracy = np.mean(np.argmax(logits, axis=1) == np.argmax(Y, axis=1))

      val_logits = model(val_images.reshape(val_images.shape[0], -1)/ 255.0)
      Yv = np.eye(10)[val_labels]
      val_loss = Loss(val_logits, Yv)
      val_accuracy = np.mean(np.argmax(val_logits, axis=1) == np.argmax(Yv, axis=1))



      print(f"End of Epoch: {epoch+1} Train Accuracy: {train_accuracy:.4f} Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.4f}")
      wandb.log({"Epoch": epoch+1, "Val Loss": val_loss, "Train Accuracy": train_accuracy, "Val Accuracy": val_accuracy})

    # ----------------------------------------------------------------------------------------------

# Sweep configuration
sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "batch_size": {"values": [48, 64, 128]},
        "hid_layers": {"values": [3, 5]},
        "nepoch": {"values": [5, 10]},
        "activation": {"values": ["relu", "tanh", "sigmoid"]},
        "init": {"values": ["Xavier", "Random"]},

    },
}
sweep_id = wandb.sweep(sweep_config, project="da6401-assignment1")
wandb.agent(sweep_id, function=train)
wandb.finish()