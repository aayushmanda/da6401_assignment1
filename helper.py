# Acknowledgement: This is inspired from Andrej Karpathy makemore: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb
# Though he uses pytorch this is solely built on numpy as external module

# -----------------------------------------------------------------------------------------------
class Linear:

  def __init__(self, fan_in, fan_out, weight_init = "Xavier", bias=True):

    self.cache = dict(x=None)
    self.gradd = dict(weight=None, bias=None)
    if weight_init == "Xavier":

      #XavierInit
      self.weight = np.random.randn(fan_in, fan_out) / (fan_in + fan_out)**0.5
      self.bias = np.zeros_like(fan_out) if bias else None
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
    self.cache = {"x": np.tanh(self.x)}
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
      return d_out * (self.out * (1 - self.out))

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
    """Cross Entropy Loss function for C class classification.

        Shapes
        ------
        Input:
            (N, C) where N is batch size and C is number of classes.
        Target:
            (N, C) where each row is a one-hot encoded vector
        Output: float
            Scalar loss.
    """

    def __init__(self, reduction='mean', eps=1e-12):  # More stable epsilon
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def __str__(self):
        return f'CrossEntropyLoss(reduction={self.reduction}, eps={self.eps})'

    def __call__(self, y, t):
        return self.forward(y, t)

    def forward(self, y, t):
        # Final layer activation is softmax and y here is logits
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        probs = exp_y / np.sum(exp_y + 1e-12, axis=1, keepdims=True)

        # print(probs)


        # Clip probabilities to [eps, 1-eps] to avoid log(0) Done this aftis after many random trials
        clipped_probs = np.clip(probs[t.astype(bool)], self.eps, 1.0 - self.eps)
        
        per_sample_loss = -np.log(clipped_probs)
        
        if self.reduction == 'mean':
            return np.mean(per_sample_loss)
        elif self.reduction == 'sum':
            return np.sum(per_sample_loss)
        else:
            return per_sample_loss

    def grad(self, y, t):
        # Simple (1/B)*(One - hot vector - yhat)
        return (1.0 / y.shape[0]) * (y - t)  # Maintain gradient scaling
# -----------------------------------------------------------------------------------------------


class Sequential:

  def __init__(self, layers):
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