import numpy as np

#let the equation be y = w*x, here let's take w as 2

#setting x and y according to the formula y = 2*x
X = np.array([1,2,3,4], dtype=np.float32)
y = np.array([-1,-2,-3,-4], dtype=np.float32)

#initializing the weight to 0.0
w = 0.0

#1. forward pass function
def forward(X, w):
    return w*X

#2. loss function, here using MSE(mean squared error)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#3. calculate gradient
# J = 1/N * (y - w*x)**2
# dJ/dw = 1/N * 2x * (y - w*x)
def gradient(X,y,y_predicted):
    return np.dot(2*X, (y_predicted - y)).mean()


###TRAINING###

#setting the learning rate
learning_rate = 0.01

#setting number of epochs, ie number of times the weights should be updated
epochs = 15

for epoch in range(epochs):

    y_pred = forward(X,w)

    l = loss(y, y_pred)

    grad = gradient(X, y, y_pred)

    w -= learning_rate*grad

    print(f"Epoch {epoch}: w = {w}")

print(y_pred)
