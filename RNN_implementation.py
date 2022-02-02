import classes_file as cls
import numpy as np

epochs = 50000
batch_size = 64
memory_duration = 3

int2binary = {}
binary_dim = 8

optimizer = cls.OptimizerAdam(learning_rate=0.1)

largest_number = pow(2, binary_dim)

binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

X = []
Y = []
for nr in range(0, epochs):

    batch_X = []
    batch_Y = []

    for i in range(0, batch_size):
        x1 = np.random.randint(255)  # we get numbers on 8 bits
        x2 = np.random.randint(255)
        y = x1 + x2

        tuple = []

        x1 = int2binary[int(x1)]
        x2 = int2binary[int(x2)]
        y = int2binary[int(y % 255)]

        tuple.append(x1)
        tuple.append(x2)

        batch_X.append(tuple)
        batch_Y.append(y)

    batch_X = np.reshape(batch_X, (8, batch_size, 2))
    batch_Y = np.reshape(batch_Y, (8, batch_size, 1))

    X.append(batch_X)
    Y.append(batch_Y)

X = np.array(X)
Y = np.array(Y)

layer1 = cls.Layer(2, 8, has_RNN=True)
activation1 = cls.ActivationSigmoid()
layer2 = cls.Layer(8, 1)
activation2 = cls.ActivationSigmoid()
loss = cls.LossMeanSquaredError()

for epoch in range(0, epochs):

    memory = [np.zeros((batch_size, 8))]

    for i in range(0, memory_duration):
        X_current = X[0][i]
        Y_current = Y[0][i]

        layer1.forward(X_current)
        activation1.forward(layer1.outputs + np.dot(memory[i], layer1.weights_h))

        memory.append(activation1.outputs)

        layer2.forward(activation1.outputs)
        activation2.forward(layer2.outputs)
        loss.forward(activation2.outputs, Y_current)

    loss.backward(activation2.outputs, Y_current)
    activation2.backward(loss.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)

    d_values = activation1.dinputs
    layer1.backward(d_values)

    dweights_h = np.dot( (memory[memory_duration]).T, d_values)
    dweights = layer1.dweights
    dbiases = layer1.dbiases

    i = memory_duration - 1

    while i > 0:
        activation1.outputs = memory[i]
        activation1.backward(d_values)
        d_values = np.dot(d_values, layer1.weights_h)
        layer1.inputs = X[0][i]
        layer1.backward(d_values)

        dweights_h += np.dot((memory[i]).T, d_values)
        dweights += layer1.dweights
        dbiases += layer1.dbiases

        i -= 1

    layer1.set_gradients(dweights,dbiases,dweights_h)

    optimizer.pre_update_params()
    optimizer.update_params(layer2)
    optimizer.update_params(layer1)

    print("Loss is: ", np.average(loss.outputs))

