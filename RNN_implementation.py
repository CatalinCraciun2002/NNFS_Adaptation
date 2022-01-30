import classes_file as cls
import numpy as np

epochs = 100
batch_size = 16

int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)

binary = np.unpackbits( np.array([range(largest_number)], dtype=np.uint8).T, axis = 1)

for i in range(largest_number):
    int2binary[i] = binary[i]

#int2binary = np.array(int2binary)

X = []
Y = []
for nr in range(0,epochs):

    batch_X = []
    batch_Y = []

    for i in range(0,batch_size):
        x1 = np.random.randint(255) # we get numbers on 8 bits
        x2 = np.random.randint(255)
        y = x1 + x2

        tuple = []

        x1 = int2binary[int(x1)]
        x2 = int2binary[int(x2)]
        y = int2binary[int(y)]

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

weights_h = np.random.randn(16,16)
memory_duration = 8


layer1 = cls.layer(2,16)
activation1 = cls.activation_sigmoid()
layer2 = cls.layer(16,1)
activation2 = cls.activation_sigmoid()

for epoch in range(0,epochs):

    memory = [np.zeros(batch_size,16)]

    for i in range(0,memory_duration):

        X_current = X[epoch][i]

        layer1.forward(X_current)
        activation1.forward(layer1.outputs + np.dot(memory[i],weights_h))
        memory.append(activation1.outputs)

        layer2.forward(activation1.outputs)
        activation2.forward(layer2.outputs)






#%%
