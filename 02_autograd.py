import torch

x = torch.rand(4, requires_grad=True)

print(x)
y = x+3   #automatically creates a computation graph
print(y)

z = y*y
z = z.mean()
print(z)

z.backward()    #dz/dx

#backward() method can only be used for scalar without passing an argument
#internally, it computes a Jacobian product

print(x.grad)


m = torch.rand(3, requires_grad=True)
print(m)
#FOR PREVENTING GRADIENT

#METHOD 1:
m.requires_grad_(False)
print(m)

#METHOD 2:
n = m.detach()  #n has same values as in m, but does not require a gradient
print(n)

#METHOD 3:
#wrap it in with statement

with torch.no_grad():
    l = m+2
    print(l)    #l tensor does not have a gradient