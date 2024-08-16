import torch
import torch.nn as nn

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction
def forward(x):
    return w*x

#training

learning_rate = 0.05
epochs = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate) #Stochastic Gradient Descent

for epoch in range (epochs):

    #prediction
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    l.backward()

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    print(f"Epoch: {epoch} Weight: {w:.3f} Loss: {l:.3f}")

print(y_pred)

#testing the prediction 
print(forward(69))