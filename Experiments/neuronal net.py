import numpy as np
from NNImplementation import BasicClasses as cls
import matplotlib.pyplot as plt

x = [
    [1,2],
    [3,4],
    [5,6]
    ]

dataset = (100 * np.random.randn(100005,100,1)) % 6 -3

layer1 = cls.Layer(1, 64)

activation1 = cls.ActivationReLU()

layer2 = cls.Layer(64, 1)

#activation2 = cls.activation_ReLU()

#layer3 = cls.layer(32 ,1)

#activation3 = cls.activation_ReLU()

#layer4 = cls.layer(64,1)

loss = cls.LossMeanSquaredError()

optimizer = cls.OptimizerAdam(learning_rate= 0.001, decay=1e-3, beta_1= 0.9, beta_2= 0.999)

#optimizer = cls.optimizer_SGD_momentum(learning_rate=0.05, decay=1e-1, momentum= 0.7)

# train epoch

for epoch in range(3000):

    x = dataset[epoch]
    y = np.sin(x)

    layer1.forward(x)
    activation1.forward( layer1.outputs )

    layer2.forward( activation1.outputs )
    #activation2.forward( layer2.outputs )

    #layer3.forward( activation2.outputs )
    #activation3.forward( layer3.outputs )

    #layer4.forward(activation3.outputs)
    loss.forward(layer2.outputs,y)


    if(epoch%500 == 0):
        print(np.mean(loss.outputs))

    loss.backward(layer2.outputs,y)
    #layer4.backward(loss.dinputs)

    #activation3.backward(layer4.dinputs)
    #layer3.backward(loss.dinputs)

    #activation2.backward(layer3.dinputs)
    layer2.backward(loss.dinputs)

    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()

    #optimizer.optimize(layer4)
    #optimizer.optimize(layer3)
    optimizer.optimize(layer2)
    optimizer.optimize(layer1)


test = np.arange(-3,3,0.1)

layer1.forward(test.reshape((-1,1)))
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)


plt.plot(test,np.sin(test))
plt.plot(test,layer2.outputs)

plt.show()
