import numpy as np

class Model:

    def __init__(self):

        self.layers = []
        self.trainable_layers = []

    def set(self, optimizer, loss, accuracy=None, loss_softmax_merge=None):

        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy
        self.loss_softmax_merge = loss_softmax_merge

    def add_layer(self, layer, trainable_layer=False):

        self.layers.append(layer)

        if trainable_layer:
            self.trainable_layers.append(layer)

    def data_shuffler(self, X, Y):

        list1 = list(range(0, len(X)))
        np.random.shuffle(list1)

        X_final = []
        Y_final = []

        for i in range(0, len(X)):
            X_final.append(X[list1[i]])
            Y_final.append(Y[list1[i]])

        X_final = np.array(X_final)
        Y_final = np.array(Y_final)

        return X_final, Y_final

    def forward(self, inputs):  # pass through all layers except loss

        nr_of_layers = len(self.layers)

        self.layers[0].forward(inputs)  # pass data through the first layer

        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].outputs)

        return self.layers[nr_of_layers - 1].outputs

    def backward(self, outputs, y):

        nr_of_layers = len(self.layers)

        if self.loss_softmax_merge is not None:

            self.loss_softmax_merge.backward(outputs, y)

            self.layers[nr_of_layers - 1].dinputs = self.loss_softmax_merge.dinputs

        else:

            self.loss.backward(outputs, y)

            self.layers[nr_of_layers - 1].backward(self.loss.dinputs)

        # now the clasification layer has already the dinputs so we start at the last desne layer

        for i in reversed(range(0, nr_of_layers - 1)):
            self.layers[i].backward(self.layers[i + 1].dinputs)

    def train(self, X, Y, batch_size, epochs, print_every=5, X_validation=None, Y_validation=None,
              regularization_loss_on=False):

        X, Y = self.data_shuffler(X, Y)

        Y = np.squeeze(Y, axis=-1)

        for epoch in range(0, epochs):

            total_loss_train = 0

            if len(X) > batch_size:

                part_length = int(len(X) / batch_size)

                if part_length * batch_size < len(X):
                    part_length += 1

                for i in range(part_length):

                    x_current = X[i * batch_size:(i + 1) * batch_size]
                    y_current = Y[i * batch_size:(i + 1) * batch_size]

                    regularization_loss = 0

                    predictions = self.forward(x_current)



                    self.loss.forward(predictions, y_current)



                    loss_train = self.loss.outputs

                    if regularization_loss_on:
                        for layer in self.trainable_layers:
                            regularization_loss += self.loss.regularization_loss(layer)

                    self.backward(predictions, y_current)

                    self.optimizer.pre_update_params()

                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)

                    total_loss_train += np.mean(loss_train) + regularization_loss

                total_loss_train /= part_length

            else:

                regularization_loss = 0

                predictions = self.forward(X)
                self.loss(predictions, Y)

                if regularization_loss_on:
                    for layer in self.trainable_layers:
                        regularization_loss += self.loss.regularization_loss(layer)

                total_loss_train = np.mean(self.loss.outputs) + regularization_loss

                self.backward(predictions, Y)

                self.optimizer.pre_update_params()

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

            loss_validation = 0
            accuracy = 0

            if X_validation is not None:

                predictions_validation = self.forward(X_validation)
                self.loss.forward(predictions_validation, Y_validation)

                regularization_loss = 0
                if regularization_loss_on:
                    for layer in self.trainable_layers:
                        regularization_loss += self.loss.regularization_loss(layer)

                loss_validation = np.mean(self.loss.outputs) + regularization_loss

                if self.accuracy is not None:
                    accuracy = self.accuracy.compare(predictions_validation, Y_validation)

            if epoch % print_every == 0:

                print("_____________________________________________")

                print("Epoch ", epoch)

                print("Current loss is: ", total_loss_train)

                if X_validation is not None:
                    print("Current validation loss is: ", loss_validation)

                if self.accuracy is not None:
                    print("Current accuracy is: ", accuracy)

                print("_____________________________________________")
