import numpy as np

#let the equation be y = w * x**2 (y equals w times x square)
#setting X and y accordingly

X = np.array([-3,-2,-1,0,1,2,3], dtype=np.float32)
y = np.array([9,4,1,0,1,4,9], dtype=np.float32)

def forward(X, w):
    return w*X**2

def loss(y_predicted,y):
    return ((y_predicted-y)**2).mean()

# J = 1/N * (w * x**2 - y)**2

# dJ.dw = 1/N * 2*x**2 * (w * x**2 - y)
 
def gradient(X,w,y_predicted):
    return np.dot(2*X**2, (y_predicted - y)).mean()

#setting the learning rate
learning_rate = 0.01

#initializing the weight to 0.0
w = 0.0

#setting number of epochs, ie number of times the weights should be updated
epochs = 15

for epoch in range(epochs):

    y_pred = forward(X,w)

    l = loss(y, y_pred)

    grad = gradient(X, y, y_pred)

    w -= learning_rate*grad

    print(f"Epoch {epoch}: w = {w}")

print(y_pred)