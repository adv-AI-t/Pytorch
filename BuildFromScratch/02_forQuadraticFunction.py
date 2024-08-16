import numpy as np
import matplotlib.pyplot as plt

#let the equation be y = w*x*x, here let's take w as 1

#setting x and y according to the formula y = x*x
X = np.array([-4,-3,-2,-1,0,1,2,3,4], dtype=np.float32)
y = np.array([16,9,4,1,0,1,4,9,16], dtype=np.float32)


#initializing the weight to 0.0
w = 0.0

#1. forward pass function
def forward(X, w):
    return w*X*X

#2. loss function, here using MSE(mean squared error)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#3. calculate gradient
# J = 1/N * (w*x*x - y)**2
# dJ/dw = 1/N * 2*x*x * (w*x*x - y)
def gradient(X,y,y_predicted):
    return np.dot(2*X*X, (y_predicted - y)).mean()


###TRAINING###

#setting the learning rate

##LEARNING RATE IS EXTREMELY IMPORTANT!!!

learning_rate = 0.0005  #perfect

# learning_rate = 0.005   #errors



#setting number of epochs, ie number of times the weights should be updated
epochs = 10

for epoch in range(epochs):

    y_pred = forward(X,w)

    l = loss(y, y_pred)

    grad = gradient(X, y, y_pred)

    w -= learning_rate*grad

    print(f"Epoch {epoch}: w = {w}")

    print(f"Epoch {epoch}: w = {w} loss = {l}")
    plt.plot(X, y_pred, color='blue')
    plt.title(f"Epoch: {epoch+1}")
    plt.xlabel("X")
    plt.ylabel("y predicted")
    plt.xlim(-5,5)
    plt.ylim(0,20)
    plt.show()

    print(y_pred)