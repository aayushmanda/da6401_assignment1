## Assignemnt 1 --- DA6401: Introduction to Deep Learning

- [Assignment Homepage](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-1--VmlldzoxMTQ2NDQwNw)

I have trid to build modules similar to pytorch!! So I have tried to use pytorch github codes and other code for reference. Dataset used are Fashion-MNIST and MNIST. And is loaded through [keras api](https://keras.io/api/datasets/fashion_mnist/).
---


### Used Python Libraries and version
This is same as one the available on Google Colab.
* python==3.11.11
* wandb==0.19.6
* tensforflow==2.18.0
* keras==3.8.0
* numpy==1.26.4
* matplotlib==3.10.0

Jump to section: [Usage](#usage)

## Implementatino of Backpropagation, Optimizers and Loss Functions

Backpropgatino Implemented for MLP layers:


Optimizers implemented:
- SGD - Stochastic Gradient Descent
- Momentum - Momentum SGD
- NAG - Nesterov Accelerated Gradient (optimized version)
- RMSProp - Root Mean Square Propagation
- Adam - Adaptive Moment Estimation
- Nadam - [Nesterov Adaptive Moment Estimation](https://cs229.stanford.edu/proj2015/054_report.pdf)

Loss functions implemented:
- Cross Entropy
- Mean Squared Error

The default values set in the file train.py are from hyperparameter tuning done using wandb sweeps.
---
 




## Usage

To run the file manually use the following command:
```sh
# This will run the default values set in train.py

$ python3 train.py -wp <wandb_project_name> -we <wandb_entity_name>
```

To run the file with custom values, check out the follwoing section.
This shows the list of all the options available and a bit of information about them.
```sh
$ python3 train.py -h
```
```
# The output of the above command is as follows:

usage: train.py [-h] -wp WANDB_PROJECT -we WANDB_ENTITY [-d DATASET] [-e EPOCHS]
                [-b BATCH_SIZE] [-l LOSS] [-o OPTIMIZER] [-lr LEARNING_RATE]
                [-m MOMENTUM] [-beta BETA] [-beta1 BETA1] [-beta2 BETA2]
                [-eps EPSILON] [-w_d WEIGHT_DECAY] [-w_i WEIGHT_INIT]
                [-nhl NUM_LAYERS] [-sz HIDDEN_SIZE] [-a ACTIVATION]

options:
  -h, --help            show this help message and exit
  -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT
                        Wandb project name
  -we WANDB_ENTITY, --wandb_entity WANDB_ENTITY
                        Wandb entity name
  -d DATASET, --dataset DATASET
                        Dataset to use choices=['fashion_mnist', 'mnist']
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -l LOSS, --loss LOSS  Loss function to use choices=['cross_entropy',
                        'mean_squared_error']
  -o OPTIMIZER, --optimizer OPTIMIZER
                        Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -m MOMENTUM, --momentum MOMENTUM
                        Momentum for Momentum and NAG
  -beta BETA, --beta BETA
                        Beta for RMSProp
  -beta1 BETA1, --beta1 BETA1
                        Beta1 for Adam and Nadam
  -beta2 BETA2, --beta2 BETA2
                        Beta2 for Adam and Nadam
  -eps EPSILON, --epsilon EPSILON
                        Epsilon for Adam and Nadam
  -w_d WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Weight decay
  -w_i WEIGHT_INIT, --weight_init WEIGHT_INIT
                        Weight initialization choices=['random', 'xavier']
  -nhl NUM_LAYERS, --num_layers NUM_LAYERS
                        Number of hidden layers
  -sz HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Hidden size
  -a ACTIVATION, --activation ACTIVATION
                        Activation function choices=['sigmoid', 'tanh', 'relu']
```

### To run a sweep using wandb:

Set the values of count and project name in sweep_code.py and then run the following command:
```sh
$ python3 sweep_code.py
```


Will update more as my assignment goes forward!!!