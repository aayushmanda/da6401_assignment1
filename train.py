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


wandb.login()

wandb.init(project="trail-1")

# Get the number of classes and their name mappings
num_classes = 10
class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
print("Done!")

##############################################################################
# Plotting a figure from each class
plt.figure(figsize=[12, 5])
img_list = []
class_list = []

for i in range(num_classes):
    position = np.argmax(train_labels==i)
    image = train_images[position,:,:]
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(class_mapping[i])
    img_list.append(image)
    class_list.append(class_mapping[i])
#api key: e6b43dd118f9a14e83fe12c597ad8d06bdfed432
wandb.log({"Images": [wandb.Image(img, caption=caption) for img, caption in zip(img_list, class_list)]})

# -----------------------------------------------------------------------------------------------
# Parse arguments and update parameters_dict
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default="trial-1", help="Wandb project name", required=True)
parser.add_argument("-we", "--wandb_entity", type=str, default="da24s016-indian-institute-of-technology-madras", help="Wandb entity name", required=True)
parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="Dataset to use choices=['fashion_mnist', 'mnist']")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Loss function to use choices=['cross_entropy', 'mean_squared_error']")
parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", help="Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for Momentum and NAG")
parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSProp")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam and Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.99, help="Beta2 for Adam and Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for Adam and Nadam")
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("-w_i", "--weight_init", type=str, default="random", help="Weight initialization choices=['random', 'xavier']")
parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Hidden size")
parser.add_argument("-a", "--activation", type=str, default="relu", help="Activation function choices=['sigmoid', 'tanh', 'relu']")

args = parser.parse_args()
# -----------------------------------------------------------------------------------------------
#Parameters
hid_layers = args.num_layers
activation = args.activation
init = args.weight_init.capitalize()
batch_size=args.batch_size
nepoch = args.epochs
opt_parameter = {"lr": args.learning_rate,
                 "optimizer": args.optimizer,
                 "momentum": args.momentum,
                 "epsilon": args.epsilon,
                 "beta": args.beta,
                 "beta1": args.beta1,
                 "beta2": args.beta2,
                 "decay": args.weight_decay}


opt = Optimizer(**opt_parameter)
loss_fn = args.loss
Loss = CrossEntropyLoss() if loss_fn=="cross_entropy" else MSE()

print(nepoch)
# -----------------------------------------------------------------------------------------------

X = train_images.reshape(train_images.shape[0], -1)/ 255.0
Y = train_labels
Y = np.eye(10)[Y]     #one_hot encoding


# -----------------------------------------------------------------------------------------------
#Model

# Define layer sizes
layer_sizes = [args.hidden_size]*(hid_layers+1)


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


else:
  raise Exception("Invalid activation function")

# -----------------------------------------------------------------------------------------------
#Training Loop


for epoch in range(nepoch):
  print("-------x-------")
  #Shuffling
  indices = np.random.permutation(X.shape[0])
  X = X[indices]
  Y = Y[indices]
  for i in range(0, train_images.shape[0], batch_size):
    Xb = X[i:i + batch_size]
    Yb = Y[i:i + batch_size]

    # Forward Pass
    logits = model(Xb)
    loss = Loss(logits, Yb)

    #Backward Pass
    dout = Loss.grad(logits, Yb)
    dout = model.backward(dout)

    batch_num = i//batch_size
    total_batch = train_images.shape[0]//batch_size

    #parameter update
    opt(model.parameters(), dout[1])



    if batch_num%200 == 0: # print every once in a while
      print(f'Epoch({epoch+1}/{nepoch})\t Batch({batch_num:2d}/{total_batch:2d}): \tTrain Loss  {loss:.4f}')
  opt.t += 1
  wandb.log({"Epoch" : epoch+1, "Train Loss": loss})



  logits = model(X)
  train_loss = Loss(logits, Y)
  train_accuracy = np.mean(np.argmax(logits, axis=1) == np.argmax(Y, axis=1))
  X_val = val_images.reshape(val_images.shape[0], -1) / 255.0


  val_logits = model(X_val)
  Yv = np.eye(10)[val_labels]
  val_loss = Loss(val_logits, Yv)
  val_accuracy = np.mean(np.argmax(val_logits, axis=1) == np.argmax(Yv, axis=1))

  print(f"End of Epoch: {epoch+1} Train Accuracy: {train_accuracy:.4f} Validation Accuracy: {val_accuracy:.4f}")
  wandb.log({"Epoch": epoch+1, "Train Accuracy": train_accuracy, "Validation Accuracy": val_accuracy})

# -----------------------------------------------------------------------------------------------
# Test Accuracy
x = test_images
x = x.reshape(x.shape[0], -1)
y = test_labels


#Forward Pass
logits = model(x)
accuracy_formula = np.mean(np.argmax(logits, axis=1) == y)
print("##################################")
print(f"Test Accuracy: {accuracy_formula}")




train_pred = np.argmax(model(X), axis=1)
val_pred = np.argmax(model(X_val), axis=1)
test_pred = np.argmax(model(x), axis=1)


x = test_images
x = x.reshape(x.shape[0], -1)/255.0
y = test_labels

#Forward Pass
logits = model(x)
accuracy_formula = np.mean(np.argmax(logits, axis=1) == y)
print(f"Test Accuracy: {accuracy_formula}")

    # Log confusion matrices
wandb.log({
    'conf_mat_train': wandb.plot.confusion_matrix(
        y_true=train_labels,
        preds=train_pred,
        class_names=class_mapping.items()),
    'conf_mat_val': wandb.plot.confusion_matrix(
        y_true=val_labels,
        preds=val_pred,
        class_names=class_mapping.items()),
    'conf_mat_test': wandb.plot.confusion_matrix(
        y_true=test_labels,
        preds=test_pred,
        class_names=class_mapping.items())
})

wandb.finish()
