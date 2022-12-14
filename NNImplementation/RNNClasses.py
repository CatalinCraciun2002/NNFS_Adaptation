import numpy as np
from NNImplementation import BasicClasses as cls


class RNNLayer(cls.Layer):

    def __init__(self, n_inputs, n_neurons, memory_duration, activation, l2_weights_regularization=0,
                 l2_biases_regularization=0):

        super().__init__(n_inputs, n_neurons, l2_weights_regularization, l2_biases_regularization)

        self.dweights_h = None
        self.deposit = None
        self.outputs = None
        self.memory = None
        self.activation = activation
        self.weights_h = 0.01 * np.random.randn(n_neurons, n_neurons)
        self.biases_h = np.zeros((1, n_neurons))
        self.memory_duration = memory_duration

    def forward_through_time(self, inputs, add_epsilon=False, epsilon=0):

        self.memory = np.zeros((self.memory_duration + 1, len(inputs[0]), len(self.weights_h)))

        if add_epsilon:
            self.weights_h[0][0] += epsilon

        for i in range(0, self.memory_duration):
            self.forward(inputs[i])

            self.outputs += np.dot(self.memory[i], self.weights_h) + self.biases_h

            # if add_epsilon and i==self.memory_duration-1:
            #    self.weights[0][0] += epsilon

            self.activation.forward(self.outputs)
            self.memory[i + 1] = self.activation.outputs

        self.outputs = self.activation.outputs

    def backward_through_time(self, dvalues, X):

        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs
        dweights_h = np.dot((self.memory[self.memory_duration - 1]).T, dvalues)

        self.backward(dvalues)

        dweights = self.dweights
        dbiases = self.dbiases

        i = self.memory_duration - 1

        while i > 0:
            dvalues = np.dot(dvalues, self.weights_h.T)

            self.activation.outputs = self.memory[i]
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs

            self.inputs = X[i - 1]
            self.backward(dvalues)

            dweights_h += np.dot((self.memory[i - 1]).T, dvalues)
            dweights += self.dweights
            dbiases += self.dbiases

            i -= 1

        dweights /= 2
        dbiases /= 2
        dweights_h /= 2

        self.set_gradients(dweights, dbiases, dweights_h)
        self.deposit = dweights_h[0][0]

    def set_gradients(self, dweights, dbiases, dweights_h):

        self.dweights = dweights

        self.dbiases = dbiases

        self.dweights_h = dweights_h


class LstmMemory:

    def __init__(self):
        self.forget_gate = None
        self.new_memory_gate = None
        self.input_gate = None
        self.out_gate = None
        self.cell_states = None
        self.outputs = None

        self.dforget_gate = None
        self.dnew_memory_gate = None
        self.dinput_gate = None
        self.dout_gate = None
        self.dcell_states = None
        self.doutputs = None


class LstmLayer(cls.Layer):

    def __init__(self, n_inputs, n_neurons, memory_duration, l2_weights_regularization=0, l2_biases_regularization=0):

        super().__init__(n_inputs, n_neurons, l2_weights_regularization, l2_biases_regularization)

        self.activation_tanh = cls.ActivationTanh()
        self.activation_sig = cls.ActivationSigmoid()

        self.weights_forget = 0.01 * np.random.randn(n_neurons, n_neurons)
        self.weights_input = 0.01 * np.random.randn(n_neurons, n_neurons)
        self.weights_new_memory = 0.01 * np.random.randn(n_neurons, n_neurons)
        self.weights_out = 0.01 * np.random.randn(n_neurons, n_neurons)

        self.dweights_forget = np.zeros_like(self.weights_forget)
        self.dweights_input = np.zeros_like(self.weights_input)
        self.dweights_new_memory = np.zeros_like(self.weights_new_memory)
        self.dweights_out = np.zeros_like(self.weights_out)

        self.memory = []
        self.memory_duration = memory_duration
        self.deposit = None

    def forward_through_time(self, inputs, add_epsilon=0, epsilon=0):

        self.memory.clear()

        if not self.memory:
            self.memory.append(LstmMemory())
            self.memory[0].outputs = np.zeros((len(inputs[0]), len(self.weights_out)))
            self.memory[0].cell_states = np.zeros((len(inputs[0]), len(self.weights_out)))

        if add_epsilon == 1:
            self.biases[0][0] += epsilon

        for i in range(0, self.memory_duration):
            self.memory.append(LstmMemory())

            self.forward(inputs[i])

            self.memory[i].forget_gate = self.outputs + np.dot(self.memory[i].outputs, self.weights_forget)
            self.memory[i].input_gate = self.outputs + np.dot(self.memory[i].outputs, self.weights_input)
            self.memory[i].new_memory_gate = self.outputs + np.dot(self.memory[i].outputs, self.weights_new_memory)
            self.memory[i].out_gate = self.outputs + np.dot(self.memory[i].outputs, self.weights_out)

            self.activation_sig.forward(self.memory[i].forget_gate)
            forget_gate = self.activation_sig.outputs

            self.activation_sig.forward(self.memory[i].input_gate)
            input_gate = self.activation_sig.outputs

            self.activation_tanh.forward(self.memory[i].new_memory_gate)
            new_memory_gate = self.activation_tanh.outputs

            self.activation_sig.forward(self.memory[i].out_gate)
            out_gate = self.activation_sig.outputs

            self.memory[i + 1].cell_states = input_gate * new_memory_gate + forget_gate * self.memory[i].cell_states

            self.activation_tanh.forward(self.memory[i + 1].cell_states)

            self.memory[i + 1].outputs = out_gate * self.activation_tanh.outputs

        self.outputs = self.memory[self.memory_duration].outputs

    def backward_through_time_init(self, dvalues):

        self.memory[-1].doutputs = dvalues

        for i in reversed(range(0, self.memory_duration)):

            # We take the derivative from the previous layer

            self.activation_tanh.forward(self.memory[i + 1].cell_states)
            self.activation_tanh.backward(self.memory[i + 1].doutputs)

            # The derivative of the cell state

            self.activation_sig.forward(self.memory[i].out_gate)
            if self.memory[i + 1].dcell_states is not None:
                self.memory[i + 1].dcell_states += self.activation_tanh.dinputs * self.activation_sig.outputs
            else:
                self.memory[i + 1].dcell_states = self.activation_tanh.dinputs * self.activation_sig.outputs
                # self.deposit = self.activation_tanh.dinputs[0][0]#self.memory[i+1].dcell_states[0][0]

            # The derivative of the out gate with the activation tanh

            self.activation_tanh.forward(self.memory[i + 1].cell_states)
            self.memory[i].dout_gate = self.activation_tanh.outputs * self.memory[i + 1].doutputs
            self.activation_sig.forward(self.memory[i].out_gate)
            self.activation_sig.backward(self.memory[i].dout_gate)
            self.memory[i].dout_gate = self.activation_sig.dinputs

            # The derivative of the new memory gate with the activation tanh

            self.activation_sig.forward(self.memory[i].input_gate)
            self.activation_tanh.forward(self.memory[i].new_memory_gate)
            self.activation_tanh.backward(self.activation_sig.outputs * self.memory[i + 1].dcell_states)
            self.memory[i].dnew_memory_gate = self.activation_tanh.dinputs

            # The derivative of the input gate with the activation sigmoid

            self.activation_tanh.forward(self.memory[i].new_memory_gate)
            self.activation_sig.forward(self.memory[i].input_gate)
            self.activation_sig.backward(self.activation_tanh.outputs * self.memory[i + 1].dcell_states)
            self.memory[i].dinput_gate = self.activation_sig.dinputs

            # The derivative of the forget gate with the activation sigmoid

            self.activation_sig.forward(self.memory[i].forget_gate)
            self.activation_sig.backward(self.memory[i + 1].cell_states * self.memory[i + 1].dcell_states)
            self.memory[i].dforget_gate = self.activation_sig.dinputs

            #  The derivative of the next state cell

            self.activation_sig.forward(self.memory[i].forget_gate)
            self.memory[i].dcell_states = self.activation_sig.outputs * self.memory[i + 1].dcell_states

            # We add the derivatives of the gates to get the derivative of the outputs of the previous time step

            self.memory[i].doutputs = np.dot(self.memory[i].dout_gate, self.weights_out.T)
            self.memory[i].doutputs += np.dot(self.memory[i].dnew_memory_gate, self.weights_new_memory.T)
            self.memory[i].doutputs += np.dot(self.memory[i].dinput_gate, self.weights_input.T)
            self.memory[i].doutputs += np.dot(self.memory[i].dforget_gate, self.weights_forget.T)

    def backward_through_time(self, dvalues, X):

        self.backward_through_time_init(dvalues)

        i = self.memory_duration - 1

        dvalues_weights_forget = self.memory[i].dforget_gate
        dvalues_weights_new_memory = self.memory[i].dnew_memory_gate
        dvalues_weights_out = self.memory[i].dout_gate
        dvalues_weights_input = self.memory[i].dinput_gate

        dweights = np.zeros_like(self.weights)
        dbiases = np.zeros_like(self.biases)

        for i in reversed(range(0, self.memory_duration)):

            if i != self.memory_duration - 1:
                dvalues_weights_forget = np.dot(dvalues_weights_forget, self.weights_forget.T) \
                                         * self.memory[i].dforget_gate
                dvalues_weights_new_memory = np.dot(dvalues_weights_new_memory, self.weights_new_memory.T) \
                                             * self.memory[i].dnew_memory_gate
                dvalues_weights_out = np.dot(dvalues_weights_out, self.weights_out.T) \
                                      * self.memory[i].dout_gate
                dvalues_weights_input = np.dot(dvalues_weights_input, self.weights_input.T) \
                                        * self.memory[i].dinput_gate

            self.dweights_forget += np.dot(self.memory[i + 1].outputs.T, dvalues_weights_forget)
            self.dweights_new_memory += np.dot(self.memory[i + 1].outputs.T, dvalues_weights_new_memory)
            self.dweights_out += np.dot(self.memory[i + 1].outputs.T, dvalues_weights_out)
            self.dweights_input += np.dot(self.memory[i + 1].outputs.T, dvalues_weights_input)

            self.inputs = X[i]

            self.backward(self.memory[i].dforget_gate)
            dweights += self.dweights
            dbiases += self.dbiases

            self.backward(self.memory[i].dnew_memory_gate)
            dweights += self.dweights
            dbiases += self.dbiases

            self.backward(self.memory[i].dout_gate)
            dweights += self.dweights
            dbiases += self.dbiases

            self.backward(self.memory[i].dinput_gate)
            dweights += self.dweights
            dbiases += self.dbiases

        self.dweights = dweights
        self.dbiases = dbiases

        self.dweights /= 2
        self.dbiases /= 2
        self.dweights_out /= 2
        self.dweights_input /= 2
        self.dweights_new_memory /= 2
        self.dweights_forget /= 2

        self.deposit = self.dbiases[0][0]


class OptimizerAdamRNN(cls.OptimizerAdam):

    def update_params(self, layer):
        super().update_params(layer)

        if not hasattr(layer, 'weight_h_cache'):
            layer.weights_h_momentum = np.zeros_like(layer.weights_h)
            layer.weights_h_cache = np.zeros_like(layer.weights_h)

        layer.weights_h_momentum = self.beta_1 * layer.weights_h_momentum + (1 - self.beta_1) * layer.dweights_h

        weights_h_momentum_corrected = layer.weights_h_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weights_h_cache = self.beta_2 * layer.weights_h_cache + (1 - self.beta_2) * layer.dweights_h ** 2

        weights_h_cache_corrected = layer.weights_h_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights_h += -self.current_learning_rate * weights_h_momentum_corrected / \
                           (np.sqrt(weights_h_cache_corrected) + self.epsilon)


class OptimizerAdamLstm(cls.OptimizerAdam):

    def update_params(self, layer):
        super().update_params(layer)

        if not hasattr(layer, 'weights_forget_cache'):
            layer.weights_forget_momentum = np.zeros_like(layer.weights_forget)
            layer.weights_forget_cache = np.zeros_like(layer.weights_forget)

            layer.weights_new_memory_momentum = np.zeros_like(layer.weights_new_memory)
            layer.weights_new_memory_cache = np.zeros_like(layer.weights_new_memory)

            layer.weights_out_momentum = np.zeros_like(layer.weights_out)
            layer.weights_out_cache = np.zeros_like(layer.weights_out)

            layer.weights_input_momentum = np.zeros_like(layer.weights_input)
            layer.weights_input_cache = np.zeros_like(layer.weights_input)

        layer.weights_forget_momentum = self.beta_1 * layer.weights_forget_momentum + (
                1 - self.beta_1) * layer.dweights_forget
        weights_forget_momentum_corrected = layer.weights_forget_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weights_forget_cache = self.beta_2 * layer.weights_forget_cache + (
                1 - self.beta_2) * layer.dweights_forget ** 2
        weights_forget_cache_corrected = layer.weights_forget_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights_forget += -self.current_learning_rate * weights_forget_momentum_corrected / \
                                (np.sqrt(weights_forget_cache_corrected) + self.epsilon)

        layer.weights_new_memory_momentum = self.beta_1 * layer.weights_new_memory_momentum + (
                1 - self.beta_1) * layer.dweights_new_memory
        weights_new_memory_momentum_corrected = layer.weights_new_memory_momentum / (
                1 - self.beta_1 ** (self.iterations + 1))
        layer.weights_new_memory_cache = self.beta_2 * layer.weights_new_memory_cache + (
                1 - self.beta_2) * layer.dweights_new_memory ** 2
        weights_new_memory_cache_corrected = layer.weights_new_memory_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights_new_memory += -self.current_learning_rate * weights_new_memory_momentum_corrected / \
                                    (np.sqrt(weights_new_memory_cache_corrected) + self.epsilon)

        layer.weights_out_momentum = self.beta_1 * layer.weights_out_momentum + (1 - self.beta_1) * layer.dweights_out
        weights_out_momentum_corrected = layer.weights_out_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weights_out_cache = self.beta_2 * layer.weights_out_cache + (1 - self.beta_2) * layer.dweights_out ** 2
        weights_out_cache_corrected = layer.weights_out_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights_out += -self.current_learning_rate * weights_out_momentum_corrected / \
                             (np.sqrt(weights_out_cache_corrected) + self.epsilon)

        layer.weights_input_momentum = self.beta_1 * layer.weights_input_momentum + (
                1 - self.beta_1) * layer.dweights_input
        weights_input_momentum_corrected = layer.weights_input_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weights_input_cache = self.beta_2 * layer.weights_input_cache + (
                1 - self.beta_2) * layer.dweights_input ** 2
        weights_input_cache_corrected = layer.weights_input_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights_input += -self.current_learning_rate * weights_input_momentum_corrected / \
                               (np.sqrt(weights_input_cache_corrected) + self.epsilon)
