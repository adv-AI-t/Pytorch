import torch

#create a weight tensor

weights = torch.rand(3, requires_grad=True)

#setting number of epochs
num_epochs = 3

for epoch in range(num_epochs):

    output = (weights*2).sum()

    #here, element-wise multiplication takes place due to weights*2
    #then, all the elements in the weights tensor are added using the sum() method
    
    print("output:", output)

    output.backward()

    print(weights.grad)

    #now, empty the gradient before going for next iteration -- IMPORTANT STEP!!!

    weights.grad.zero_()


    #USING THE OPTIMIZER

    weights = torch.rand(3, requires_grad=True)

    optimizer = torch.optim.SGD(weights, lr = 0.1)      #first pass the tensor, then specify the learning rate with lr = 

    #SGD is Stochastic Gradient Descent, there are other optimizers available too like Adam, Adagrad, etc

    optimizer.step()    #perform the optimization step

    optimizer.zero_grad()   #reset the gradients
    
