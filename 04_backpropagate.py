import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)   #we will be calculating the gradient of this w tensor

#forward pass:
y_pred = w*x
 
#compute the loss:
loss = (y - y_pred)**2
print(loss)

#backward pass:
loss.backward()

#print the first gradient
print(w.grad)