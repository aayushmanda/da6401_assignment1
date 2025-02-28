# Acknowledgement: This is inspired from Andrej Karpathy makemore: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb
# Though he uses pytorch this is solely built on numpy as external module
import numpy as np

# -----------------------------------------------------------------------------------------------
class Linear:

  def __init__(self, fan_in, fan_out, weight_init = "Xavier", bias=True):

    self.cache = dict(x=None)
    self.gradd = dict(weight=None, bias=None)
    if weight_init == "Xavier":
      #XavierInit
      self.weight = np.random.randn(fan_in, fan_out) / (fan_in + fan_out)**0.5
      self.bias = np.zeros(fan_out, dtype="f") if bias else None
    else:
      #RandomInit
      self.weight = np.random.randn(fan_in, fan_out)
      self.bias = np.random.randn(fan_out) if bias else None

  def __str__(self):
      return "Linear({:d}, {:d}, bias={})".format(self.weight.shape[0], self.weight.shape[1], self.bias is not None)


  def __call__(self, x):
      self.out = x @ self.weight
      if self.bias is not None:
          self.out += self.bias
      # Store input for backward pass
      self.cache["x"] = x
      return self.out

  def grad(self, d_out):
    x = self.cache["x"]
    # Weight gradient: x^T @ d_out
    self.gradd["weight"] = x.T @ d_out

    dzdx = d_out@self.weight.T

    # Bias gradient: sum over batch
    if self.bias is not None:
        self.gradd["bias"] = np.sum(d_out, axis=0)

    # returning dzdx
    return dzdx

  def parameters(self):
    return [self.weight] + ([self.bias] if self.bias is not None else [])



# -----------------------------------------------------------------------------------------------

class Tanh:
  def __call__(self, x):
    self.x = x
    self.out = np.tanh(self.x)
    # self.cache = {"x": np.tanh(self.x)}
    return self.out
  def parameters(self):
    # Activation Function
    return []

  def grad(self, dout):
    return dout * (1 - self.out**2)


# -----------------------------------------------------------------------------------------------

class Sigmoid:
  def __call__(self, x):
      self.out = 1 / (1 + np.exp(-x))
      return self.out

  def grad(self, d_out):
      # sigmoid derivative: σ(x)(1 - σ(x))
      return (self.out * (1 - self.out)) * d_out

  def parameters(self):
      return []



# -----------------------------------------------------------------------------------------------


class Relu:
  def __call__(self, x):
      self.cache = {"x": x}
      self.out = np.maximum(0, x)
      return self.out

  def grad(self, d_out):
      x = self.cache["x"]
      dx = np.ones_like(x)
      dx[x < 0] = 0
      return np.array(d_out) * dx

  def parameters(self):
      return []


# -----------------------------------------------------------------------------------------------

class CrossEntropyLoss:

    def __init__(self, reduction='mean', eps=1e-12):  # More stable epsilon
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def __str__(self):
        return f'CrossEntropyLoss(reduction={self.reduction}, eps={self.eps})'

    def __call__(self, y, y_true):
        return self.forward(y, y_true)

    def forward(self, y, y_true):
        # Final layer activation is softmax and y here is logits
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        probs = exp_y / np.sum(exp_y + 1e-12, axis=1, keepdims=True)

        # print(probs)


        # Clip probabilities to [eps, 1-eps] to avoid log(0) Done this aftis after many random trials
        clipped_probs = np.clip(probs[y_true.astype(bool)], self.eps, 1.0 - self.eps)

        per_sample_loss = -np.log(clipped_probs)

        if self.reduction == 'mean':
            return np.mean(per_sample_loss)
        elif self.reduction == 'sum':
            return np.sum(per_sample_loss)
        else:
            return per_sample_loss

    def grad(self, y, y_true):
        # Simple (1/B)*(One - hot vector - yhat)
        return (1.0 / y.shape[0]) * (y - y_true)  # Maintain gradient scaling

# -----------------------------------------------------------------------------------------------

class MSE:
  def __init__(self, eps=1e-8):
      self.eps = eps
  def __call__(self, y, y_true):
      return self.forward(y, y_true)

  def forward(self, y, y_true):
      exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
      probs = exp_y / np.sum(exp_y + 1e-12, axis=1, keepdims=True)
      # Clip probabilities to [eps, 1-eps] to avoid log(0) Done this aftis after many random trials
      # clipped_probs = np.clip(probs[y_true.astype(bool)], self.eps, 1.0 - self.eps)
      return np.mean((probs - y_true)**2)
    
  def grad(self, y, y_true):
      # y: logits with shape (batch_size, num_classes)
      # Compute softmax probabilities
      exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
      probs = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + 1e-12)
      batch_size, num_classes = y.shape
      grad_input = np.zeros_like(y)
      # For each sample in the batch, compute:
      #   dL/dp = 2*(probs - y_true)/batch_size   (MSE derivative w.r.t. softmax outputs)
      #   dp/dz = Jacobian of softmax = diag(p) - p pᵀ
      #   and then dL/dz = (dp/dz) · (dL/dp)
      #   Loop here makes we iterate through every x_i from batch
      for i in range(batch_size):
          p = probs[i].reshape(-1, 1)  # Column vector (num_classes, 1)
          # Jacobian for softmax (num_classes x num_classes)
          J = np.diagflat(p) - np.dot(p, p.T) #outerproduct
          dLdp = 2 * (probs[i] - y_true[i]) / batch_size
          grad_input[i, :] = np.dot(J, dLdp)
      return grad_input

# -----------------------------------------------------------------------------------------------

class Sequential:

  def __init__(self, layers=None):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out

  def append(self, layer):
    self.layers.append(layer)

  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]

  def backward(self, d_out):
    # Backpropagate through layers in reverse order
    d = d_out
    gradients = []
    #storing layer.weight, layer.bias grad in a list also

    for layer in reversed(self.layers):
        d = layer.grad(d)
        if hasattr(layer, 'weight'):
            gradients.append(layer.gradd["bias"] if layer.bias is not None else None)
            gradients.append(layer.gradd["weight"])
    return d, list(reversed(gradients))

# -----------------------------------------------------------------------------------------------

class Optimizer():
    def __init__(self, lr=0.001, optimizer="sgd", momentum=0.9,
                 epsilon=1e-8, beta=0.9, beta1=0.9, beta2=0.999, t=0, decay=0):
      self.lr = lr
      self.optimizer = optimizer
      self.momentum = momentum
      self.epsilon = epsilon
      self.beta = beta
      self.beta1 = beta1
      self.beta2 = beta2
      self.t = t
      self.decay = decay
      self.velocity = None
      self.moments = None
      
    def __call__(self, param, dparam):
      self.t+=1
      self.run(param, dparam)

    def run(self, param, dparam):
        if(self.optimizer == "sgd"):
            self.SGD(param, dparam)
        elif(self.optimizer == "momentum"):
            self.MomentumGD(param, dparam)
        elif(self.optimizer == "nag"):
            self.NAG(param, dparam)
        elif(self.optimizer == "rmsprop"):
            self.RMSProp(param, dparam)
        elif(self.optimizer == "adam"):
            self.Adam(param, dparam)
        elif (self.optimizer == "nadam"):
            self.NAdam(param, dparam)
        else:
            raise Exception("Invalid optimizer")

    def SGD(self, param, dparam):
        clip_value = 1e-2
        for p, grad in zip(param, dparam):
            clipped_dparam = np.clip(grad, -clip_value, clip_value)
            p -= self.lr * clipped_dparam #grad

    def MomentumGD(self, param, dparam):
        clip_value = 1e-3
        #tried using zeros but got Value error maximum dim support for ndarray is 32.
        self.velocity = [np.zeros_like(p) for p in param]
        for i, (u, param, grad) in enumerate(zip(self.velocity, param, dparam)):
            clipped_dparam = np.clip(grad, -clip_value, clip_value)
            u = self.momentum * u + 0.1 * clipped_dparam
            param -= self.lr * u     #clipped_dparam #grad

    def NAG(self, param, dparam):
        clip_value = 1e-3
        self.velocity = [np.zeros_like(p) for p in param]
        for i, (u, param, grad) in enumerate(zip(self.velocity, param, dparam)):
            clipped_dparam = np.clip(grad, -clip_value, clip_value)
            u = self.momentum * u + 0.1 * clipped_dparam
            param -= self.lr * u + clipped_dparam    #clipped_dparam #grad

    def RMSProp(self, param, dparam ):
        clip_value = 1e-3
        self.velocity = [np.zeros_like(p) for p in param]
        for i, (u, param, grad) in enumerate(zip(self.velocity, param, dparam)):
            clipped_grad = np.clip(grad, -clip_value, clip_value)
            u = self.beta * u + (1 - self.beta) * (clipped_grad**2)
            param -= ((self.lr * clipped_grad) / (np.sqrt(u + self.epsilon)))

    def Adam(self, param, dparam):
        self.moments =  [np.zeros_like(p) for p in param]
        self.velocity =  [np.zeros_like(p) for p in param]
        for i, (m, v, param, grad) in enumerate(zip(self.moments, self.velocity, param, dparam)):
            m = self.beta1 * m + (1 - self.beta1) * grad
            m_hat = m/(1-self.beta1)

            v = self.beta1 * v + (1 - self.beta2) * (grad**2)
            v_hat = v/(1-self.beta2)

            param -= ((self.lr * m_hat) / (np.sqrt(v_hat + self.epsilon)))


    def NAdam(self, param, dparam, epoch):
        i = epoch
        self.moments =  [np.zeros_like(p) for p in param]
        self.velocity =  [np.zeros_like(p) for p in param]
        for i, (m, v, param, grad) in enumerate(zip(self.moments, self.velocity, param, dparam)):
            m = self.beta1 * m + (1 - self.beta1) * grad
            m_hat = m/(1-self.beta1**(i+1))

            v = self.beta1 * v + (1 - self.beta2) * (grad**2)
            v_hat = v/(1-self.beta2**(i+1))

            param -= (self.lr  / (np.sqrt(v_hat + self.epsilon))) * (self.beta1*m_hat + (1-self.beta1)*grad/(1-self.beta1**(i+1)))
