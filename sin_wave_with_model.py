import classes_file as cls
import numpy as np
import model

X = np.random.randn(10000,1) % np.pi - np.pi/2
Y = np.sin(X)

X_validation = np.random.rand(100,1) % np.pi - np.pi/2
Y_validation = np.sin(X_validation)

model = model.Model()

model.add_layer(cls.layer(1,64),trainable_layer=True)
model.add_layer(cls.activation_ReLU())
model.add_layer(cls.layer(64,1),trainable_layer=True)

model.set(cls.optimizer_Adam(learning_rate=0.01), loss=cls.loss_mean_squared_error())

model.train(X,Y,64,1000,5, X_validation=X_validation, Y_validation=Y_validation)
