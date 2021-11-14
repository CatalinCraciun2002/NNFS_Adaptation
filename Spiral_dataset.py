import numpy as np
import classes_file as cls
import model
from nnfs.datasets import spiral_data

X,Y = spiral_data(1000, 3)

model = model.Model()

model.add_layer(cls.layer(2,512,l2_biases_regularization=5e-4, l2_weights_regularization= 5e-4),trainable_layer=True)
model.add_layer(cls.activation_ReLU())
#model.add_layer(cls.layer(128,64,l2_biases_regularization=5e-4, l2_weights_regularization= 5e-4),trainable_layer=True)
#model.add_layer(cls.activation_ReLU())
model.add_layer(cls.layer(512,3),trainable_layer=True)
model.add_layer(cls.ActivationSoftmax())

model.set(optimizer=cls.optimizer_Adam(0.02, decay=5e-4, beta_2=0.999),
          loss= cls.CategoricalCrossEntropyLossFunction(),
          accuracy=cls.AccuracyCrossEntropy(),
          loss_softmax_merge=cls.CategoricalCrossEntropySoftmaxActivation())

X,Y = model.data_shuffler(X,Y)

length = int(len(X)/8*5)

X_train = X[:length]
Y_train = Y[:length]
X_valid = X[length:]
Y_valid = Y[length:]

model.train(X=X_train, Y=Y_train, batch_size= 64, epochs=10000,
            print_every=50, X_validation=X_valid,
            Y_validation=Y_valid, regulariztion_loss_on=True)