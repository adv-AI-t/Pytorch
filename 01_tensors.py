import torch

#first paramenter is dimension, second is size
#empty creates a tensor initiated with 0. values

myTensor = torch.empty(2,5)     #create an empty tensor of dimension 2 and size 5
print(myTensor)


#ones method creates a tensor of values 1.
onesTensor = torch.ones(3,4)
print(onesTensor)

#3rd parameter is to set datatype
newTensor = torch.ones(2,2, dtype = torch.int)
print(newTensor)

#tensor method is used to create tensor from a list
lstTensor = torch.tensor([6,9,69])
print(lstTensor)


#OPERATIONS ON TENSORS

x = torch.rand(3,3)
y = torch.rand(3,3)
z = x + y           #alternately use z = torch.add(x,y)
print(z)


#FOR INPLACE OPERATION, there is a underscore (_) after method name

print(y)
y.add_(x)
#y tensor will be updated after adding x to it
print(y)

#SLICING

print(y[:,0])   #: denotes we are selecting all the rows, 0 denotes that we are selecting the 0th column
print(y[2, :])  #: denotes we are selecting all the columns, 1 denotes we are selecting the 1st row


#RESHAPING THE TENSOR

tens = torch.rand(4,4)
print(tens)
y = tens.view(-1,8) #here we are specifying that 
print(y.size())

#CONVERT TENSOR TO NUMPY ARRAY

import numpy as np

a = torch.zeros(6)
print(a)
b = a.numpy()
print(b)

#NOTE THAT is both are stored on GPU, then is I modify the tensor, the array too will be modified as they point to the same memory location

a.add_(1)
print(b)

#NUMPY ARRAY TO TENSOR

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

#WHAT IS REQURES_GRAD??

#requires_grad parameter when set to True denotes that further in the process, we will be requiring the calculate the gradient of the tensor

x = torch.ones(5, requires_grad = True)
print(x)