import cv2
import os
from NNImplementation import BasicClasses as cls, ModelClassIntegration

image_data = cv2.imread('../fashion_mnist_images/train/7/0002.png',
                        cv2.IMREAD_UNCHANGED)

import numpy as np
np.set_printoptions(linewidth=200)

X_train=[]
Y_train=[]

for files in os.listdir('../fashion_mnist_images/train'):
    for image in os.listdir('fashion_mnist_images/train/' + files):
        x = np.array(cv2.imread('fashion_mnist_images/train/' + files + '/' + image, cv2.IMREAD_UNCHANGED))
        x = x.reshape(28*28)
        x = (x - 127.5) / 127.5

        X_train.append(x)
        Y_train.append(files)

print(np.max(X_train[1]))

X_test=[]
Y_test=[]

for files in os.listdir('../fashion_mnist_images/test'):
    for image in os.listdir('fashion_mnist_images/test/' + files):
        x = np.array(cv2.imread('fashion_mnist_images/test/' + files + '/' + image, cv2.IMREAD_UNCHANGED))
        x = x.reshape(28*28)
        x = (x - 127.5) / 127.5


        X_test.append(x)
        Y_test.append(files)


model = model.Model()

X_train, Y_train = model.data_shuffler(X_train,Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

model.add_layer(cls.Layer(28 * 28, 128, l2_biases_regularization=1e-3, l2_weights_regularization=1e-3), trainable_layer=True)
model.add_layer(cls.ActivationReLU())
model.add_layer(cls.Layer(128, 64, l2_biases_regularization=1e-3, l2_weights_regularization=1e-3), trainable_layer=True)
model.add_layer(cls.ActivationReLU())
model.add_layer(cls.Layer(64, 10), trainable_layer=True)
model.add_layer(cls.ActivationSoftmax())


model.set(optimizer=cls.OptimizerAdam(learning_rate=0.005, decay=5e-4),
          loss=cls.CategoricalCrossEntropyLossFunction(),
          loss_softmax_merge=cls.CategoricalCrossEntropySoftmaxActivation()
          )

print("Model started training")


model.train(X_train, Y_train, 128, 1000, print_every=50, X_validation=X_test, Y_validation=Y_test,
            regularization_loss_on=True)







