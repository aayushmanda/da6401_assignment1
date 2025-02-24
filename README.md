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

### Arguments to be Supported

| Name              | Default Value  | Description                                                                 |
|-------------------|----------------|-----------------------------------------------------------------------------|
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard.      |
| `-we`, `--wandb_entity`  | myname        | WandB Entity used to track experiments in the Weights & Biases dashboard.  |
| `-d`, `--dataset`         | fashion_mnist | choices: ["mnist", "fashion_mnist"]                                        |
| `-e`, `--epochs`          | 1            | Number of epochs to train the neural network.                              |
| `-b`, `--batch_size`      | 4            | Batch size used to train the neural network.                               |
| `-l`, `--loss`            | cross_entropy| choices: ["mean_squared_error", "cross_entropy"]                           |
| `-o`, `--optimizer`       | sgd          | choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]            |
| `-lr`, `--learning_rate`  | 0.1          | Learning rate used to optimize model parameters.                           |
| `-m`, `--momentum`        | 0.5          | Momentum used by momentum and nag optimizers.                              |
| `-beta`, `--beta`         | 0.5          | Beta used by rmsprop optimizer.                                            |
| `-beta1`, `--beta1`       | 0.5          | Beta1 used by adam and nadam optimizers.                                   |
| `-beta2`, `--beta2`       | 0.5          | Beta2 used by adam and nadam optimizers.                                   |
| `-eps`, `--epsilon`       | 0.000001     | Epsilon used by optimizers.                                                |
| `-w_d`, `--weight_decay`  | 0.0          | Weight decay used by optimizers.                                           |
| `-w_i`, `--weight_init`   | random       | choices: ["random", "Xavier"]                                              |
| `-nhl`, `--num_layers`    | 1            | Number of hidden layers used in feedforward neural network.                |
| `-sz`, `--hidden_size`    | 4            | Number of hidden neurons in a feedforward layer.                           |
| `-a`, `--activation`      | sigmoid      | choices: ["identity", "sigmoid", "tanh", "ReLU"]                           |

---

### To run a sweep using wandb:

Set the values of count and project name in sweep_code.py and then run the following command:
```sh
$ python3 sweep_code.py
```


Will update more as my assignment goes forward!!!