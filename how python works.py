import numpy as np
import matplotlib.pyplot as plt
#import classes_file

x = np.array([
    [1,-2,3],
    [4,-5,6],
    [7,-8,9]
])
'''
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = max(x[i][j], 0)



activation = cls.activation_ReLU()

activation.forward(x)

loss = cls.custom_loss_abs()

loss.forward(x[0],x[1])

loss.backward()

#dataset = np.random.randn(10,10,1)

#print(dataset)

#print(np.mean(x,axis=1))
test = np.arange(-3,3,0.1)

print(x.reshape(9))

plt.plot(test,np.sin(test))



class JustAClass:

    def __init__(self, val):

        self.val= val

    def change_val(self):

        self.val = "change"
        print("done")


def function(class_val, change_function):

    change_function()

    print(class_val[0])



cls = JustAClass("original")

function([cls.val], cls.change_val)
'''


for i in range(1,0,-1):
    print(i)
