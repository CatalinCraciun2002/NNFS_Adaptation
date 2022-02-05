import classes_file as cls
import numpy as np


epochs = 1000
batch_size = 16
memory_duration = 3
layer1_neurons = 16

int2binary = {}
binary_dim = 8

optimizer = cls.OptimizerAdam(learning_rate=0.1, decay=1e-5)

largest_number = pow(2, binary_dim)

binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

X = []
Y = []
for nr in range(0, epochs):

    batch_X = []
    batch_Y = []

    for i in range(0,8):
        batch_X.append([])
        batch_Y.append([])

    for i in range(0, batch_size):
        x1 = np.random.randint(255)  # we get numbers on 8 bits
        x2 = np.random.randint(255)
        y = (x1 + x2) % 256

        x1 = int2binary[int(x1)]
        x2 = int2binary[int(x2)]
        y = int2binary[int(y)]

        for i in range(0,8):
            batch_X[i].append([ x1[7-i], x2[7-i] ])
            batch_Y[i].append([y[7-i]])



    X.append(batch_X)
    Y.append(batch_Y)

X = np.array(X)
Y = np.array(Y)

layer1 = cls.RNNLayer(2, layer1_neurons, activation=cls.ActivationSigmoid(), memory_duration=memory_duration)

layer2 = cls.Layer(layer1_neurons, 1)
activation2 = cls.ActivationSigmoid()
loss = cls.LossMeanSquaredError()

def forward(memory_duration, X_current, Y_current, empty_memory=False):

    layer1.forward_RNN(X_current,empty_memory=empty_memory)
    layer2.forward(layer1.outputs)
    activation2.forward(layer2.outputs)
    loss.forward(activation2.outputs, Y_current[memory_duration-1])

def backward(memory_duration, Y_current):

    loss.backward(activation2.outputs, Y_current[memory_duration - 1])
    activation2.backward(loss.dinputs)
    layer2.backward(activation2.dinputs)
    layer1.backward_RNN(layer2.dinputs, X[epoch])


checker = cls.GradientChecker()

for epoch_fin in range(0, epochs*100):

    epoch = 0

    memory = [np.zeros((batch_size, layer1_neurons))]

    X_current = X[epoch]
    Y_current = Y[epoch]

    forward(memory_duration, X_current, Y_current)
    backward(memory_duration, Y_current)

    #checker.check_weights_h(layer=layer1, loss=loss, forward=forward_legacy, backward=backward_legacy, X=X_current, Y=Y_current,
     #                  memory=memory,  memory_duration=memory_duration, print_out=True)


    optimizer.pre_update_params()
    optimizer.update_params(layer2)
    optimizer.update_params(layer1)

    print("Loss is: ", np.average(loss.outputs))
