import numpy as np

class Model:

    def __init__(self):

        self.layers = []
        self.trainable_layers = []


    def set(self, optimizer, loss, accuracy =None, loss_softmax_merge=None):

        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy
        self.loss_softmax_merge = loss_softmax_merge


    def add_layer(self, layer, trainable_layer=False):

        self.layers.append(layer)

        if trainable_layer :
            self.trainable_layers.append(layer)

    def data_shuffler(self, X, Y):

        list1 = list(range(0,len(X)))
        np.random.shuffle(list1)

        X_final = []
        Y_final = []

        for i in range(0,len(X)):
            X_final.append(X[list1[i]])
            Y_final.append(Y[list1[i]])

        X_final = np.array(X_final)
        Y_final = np.array(Y_final)

        return X_final, Y_final


    def forward(self, inputs): #pass through all layers except loss

        nr_of_layers = len(self.layers)

        self.layers[0].forward(inputs)  # pass data through the first layer

        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].outputs)


        return self.layers[nr_of_layers-1].outputs


    def backward(self, predictions, y):

        nr_of_layers = len(self.layers)

        if self.loss_softmax_merge is not None:

            self.loss_softmax_merge.backward(predictions, y)

            self.layers[nr_of_layers-1].dinputs = self.loss_softmax_merge.dinputs

        else:

            self.loss.backward(predictions, y)

            self.layers[nr_of_layers-1].backward(self.loss.dinputs)


        # now the clasification layer has already the dinputs so we start at the last desne layer

        for i in reversed(range(0,nr_of_layers-1)):
            self.layers[i].backward(self.layers[i + 1].dinputs)


    def train(self, X, Y, batch_size, epochs, print_every=5, X_validation= None, Y_validation = None, regulariztion_loss_on = False):

        X, Y = self.data_shuffler(X, Y)

        for epoch in range(0,epochs):

             if len(X) > batch_size:
                part_length = int(len(X) / batch_size)

                #  if part_length * batch_size < len(X) :
                #     part_length += 1

                total_loss = 0
                total_accuracy = 0

                for i in range(part_length):

                    X_current = X[i * batch_size:(i + 1) * batch_size]
                    Y_current = Y[i * batch_size:(i + 1) * batch_size]

                    #X_current = X_current.shape(-1,self.trainable_layers[0].neurons)
                    #Y_current = Y_current.shape(-1,self.trainable_layers[-1].neurons)
                    #X_current = X_current.reshape(batch_size,-1)
                    #Y_current = Y_current.reshape(batch_size,-1)

                    accuracy = 0
                    predictions_verification = None
                    loss_verification = -1
                    regularization_loss = 0

                    if X_validation is not None:
                        predictions_verification = self.forward(X_validation)
                        self.loss.forward(predictions_verification,Y_validation)
                        loss_verification = np.mean(self.loss.outputs)


                    predictions = self.forward(X_current)
                    self.loss.forward(predictions, Y_current)

                    if regulariztion_loss_on:
                        for layer in self.trainable_layers:
                            regularization_loss+= self.loss.regularization_loss(layer)


                    if self.accuracy is not None and X_validation is not None:
                        accuracy = self.accuracy.compare(predictions_verification, Y_validation)


                    self.backward(predictions,Y_current)

                    self.optimizer.pre_update_params()

                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)

                    total_loss = loss_verification + regularization_loss
                    total_accuracy = accuracy


                    if ((epoch+1) * i) % print_every == 0:
                        print("Current loss at iteration ", i*(epoch+1) ," is: ", total_loss)
                        print("Loss from training is: ", np.mean(self.loss.outputs) + regularization_loss)
                        print("learning rate is: ", self.optimizer.current_learning_rate)
                        if self.accuracy is not None:
                            print("Current accuracy is: ", total_accuracy)

             else:

                predictions = self.forward(X)

                self.loss(predictions, Y)

                if self.accuracy is not None:
                    accuracy = self.accuracy.compare(predictions, Y)

                loss = np.mean(self.loss.outputs)

                self.backward(predictions, Y)

                self.optimizer.pre_update_params()

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

                if(epoch % print_every == 0):
                    print("Current loss is: ", loss)
                    if self.accuracy is not None:
                        print("Current accuracy is: ", accuracy)

