import numpy as np

class Layer:

    def __init__(self, n_inputs, n_neurons, l2_weights_regularization=0, l2_biases_regularization=0):

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l2_weights_regularization = l2_weights_regularization
        self.l2_biases_regularization = l2_biases_regularization

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):

        self.inputs = inputs

        self.outputs = np.dot(inputs, self.weights)

        self.outputs += self.biases

    def backward(self, dvalues):

        self.dinputs = np.dot(dvalues, self.weights.T)

        self.dweights = np.dot(self.inputs.T, dvalues)

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if (self.l2_weights_regularization != 0):
            self.dweights += 2 * self.l2_weights_regularization * self.weights

        if (self.l2_biases_regularization != 0):
            self.dbiases += 2 * self.l2_biases_regularization * self.biases


class AccuracyCrossEntropy:

    def compare(self, predictions, targets):

        predictions = np.argmax(predictions, axis=1)

        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        total = len(predictions)
        counter = 0

        for i in predictions:
            if predictions[i] == targets[i]:
                counter += 1

        return counter / total


class ActivationReLU:

    def forward(self, inputs):

        self.inputs = inputs

        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        for i in range(len(dvalues)):
            for j in range(len(dvalues[i])):
                if self.inputs[i][j] < 0:
                    self.dinputs[i][j] = 0


class ActivationSigmoid:

    def forward(self, inputs):
        self.inputs = inputs

        self.outputs = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = self.outputs * (1 - self.outputs) * dvalues


class ActivationTanh:

    def forward(self, inputs):
        self.inputs = inputs

        plus_e = np.exp(inputs)
        minus_e = np.exp(-inputs)

        self.outputs = (plus_e - minus_e) / (plus_e + minus_e)

        if np.isnan(np.average(self.outputs)):
            return

    def backward(self, dvalues):
        self.dinputs = (1 - np.square(self.outputs)) * dvalues


class LossAbs:

    def forward(self, inputs, targets):

        self.inputs = inputs
        self.targets = targets
        self.outputs = abs(inputs - targets)

    def backward(self, batch_size):

        self.dinputs = []

        for i, j in zip(self.inputs, self.targets):
            if i - j > 0:
                self.dinputs.append([1])
            else:
                self.dinputs.append([-1])

        self.dinputs = np.array(self.dinputs)

        self.dinputs = self.dinputs / batch_size


class LossMeanSquaredError:

    def forward(self, inputs, targets):

        self.outputs = (inputs - targets) * (inputs - targets)

    def backward(self, inputs, targets):

        self.dinputs = 2 * (inputs - targets) / len(inputs)

    def regularization_loss(self, layer):

        regularization_loss = 0

        if layer.l2_weight_regularization > 0:
            regularization_loss += layer.l2_weight_regularization * np.sum(layer.weights * layer.weights)

        if layer.l2_bias_regularization > 0:
            regularization_loss += layer.l2_bias_regularization * np.sum(layer.biases * layer.biases)

        return regularization_loss


class OptimizerSgdMomentum:

    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.curent_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0

    def pre_update_params(self):
        self.curent_learning_rate = self.learning_rate / (1 + self.decay * self.iteration)

        self.iteration += 1

    def optimize(self, layer):
        if not hasattr(layer, 'weights_update'):
            layer.weights_update = np.zeros_like(layer.weights)
            layer.biases_update = np.zeros_like(layer.biases)

        layer.weights_update = self.momentum * layer.weights_update - self.curent_learning_rate * layer.dweights
        layer.biases_update = self.momentum * layer.biases_update - self.curent_learning_rate * layer.dbiases

        layer.weights += layer.weights_update
        layer.biases += layer.biases_update


class OptimizerAdam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta_1 * layer.weight_momentum + (1 - self.beta_1) * layer.dweights

        layer.bias_momentum = self.beta_1 * layer.bias_momentum + (1 - self.beta_1) * layer.dbiases

        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentum_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentum_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)


class CategoricalCrossEntropySoftmaxActivation:

    def __init__(self):

        self.loss = CategoricalCrossEntropyLossFunction()
        self.activation = ActivationSoftmax()

    def forward(self, inputs, target_values):

        self.activation.forward(inputs)
        self.predictions = self.activation.outputs
        self.loss.forward(self.predictions, target_values)
        self.output = self.loss.outputs

    def backward(self, inputs, target_values):

        samples = len(inputs)

        if len(target_values.shape) == 2:
            target_values = np.argmax(target_values, axis=1)

        self.dinputs = inputs.copy()  # these are the predicted values of the softmax function

        for i, j in zip(range(samples), target_values):
            self.dinputs[i][int(j)] -= 1

        self.dinputs /= samples


class CategoricalCrossEntropyLossFunction:

    def forward(self, inputs, target_values):

        if len(target_values.shape) == 2:
            target_values = np.argmax(target_values, axis=1)

        self.outputs = []

        for i, j in zip(range(len(inputs)), target_values):
            self.outputs.append(inputs[i][int(j)])

        self.outputs = np.array(self.outputs)

        self.outputs = np.clip(self.outputs, 1e-7, 1 - 1e-7)

        self.outputs = -np.log(self.outputs)

    def regularization_loss(self, layer):

        regularization_loss = 0

        if layer.l2_weights_regularization > 0:
            regularization_loss += layer.l2_weights_regularization * np.sum(layer.weights * layer.weights)

        if layer.l2_biases_regularization > 0:
            regularization_loss += layer.l2_biases_regularization * np.sum(layer.biases * layer.biases)

        return regularization_loss


class ActivationSoftmax:

    def forward(self, inputs):
        exp_value = inputs

        exp_value -= np.max(inputs, axis=1, keepdims=True)

        exp_value = np.exp(exp_value)

        self.outputs = exp_value / np.sum(exp_value, axis=1, keepdims=True)


class GradientChecker:

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.derivative = 0

    def check_custom(self, loss, forward, backward, X, Y, memory_duration=1, print_out=True):

        forward(memory_duration, X, Y)
        self.derivative = backward(memory_duration, X, Y)

        forward(memory_duration, X, Y, empty_memory=True, add_epsilon=True, epsilon=self.epsilon)

        result_epsilon_plus = loss.outputs

        forward(memory_duration, X, Y, empty_memory=True, add_epsilon=True, epsilon=-self.epsilon)

        result_epsilon_minus = loss.outputs

        if (print_out):
            print(self.derivative, np.average((result_epsilon_plus - result_epsilon_minus) / (2 * self.epsilon)))

    def check_attribute(self,attribute, layer, loss, forward, backward, X, Y, memory_duration=1, print_out=True):

        forward(memory_duration, X, Y)
        backward(memory_duration, X, Y)

        derivative = (layer.__getattribute__("d" + attribute))[0][0]

        (layer.__getattribute__(attribute))[0][0] += self.epsilon
        forward(memory_duration, X, Y, empty_memory=True)

        result_epsilon_plus = loss.outputs

        (layer.__getattribute__(attribute))[0][0] -= 2 * self.epsilon
        forward(memory_duration, X, Y, empty_memory=True)

        result_epsilon_minus = loss.outputs

        if print_out:
            print(derivative, np.average((result_epsilon_plus - result_epsilon_minus) / (2 * self.epsilon)))

        (layer.__getattribute__(attribute))[0][0] += self.epsilon
