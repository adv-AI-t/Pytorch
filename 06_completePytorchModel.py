import torch
import torch.nn as nn

#pay attention to the shape...it is a 2d array

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)    
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

#here there are 4 samples and 1 feature

model = nn.Linear(input_size, output_size)

print(f"Prediction before training: {model(X_test).item():.3f}")

#training

learning_rate = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #Stochastic Gradient Descent

for epoch in range (epochs):

    #prediction
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    l.backward()

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    # print(f"Epoch: {epoch} Weight: {w:.3f} Loss: {l:.3f}")

print(y_pred)

print(f"Prediction after training: {model(X_test).item():.3f}")