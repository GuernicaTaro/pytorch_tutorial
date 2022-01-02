import torch

x = torch.tensor([4.,1.], requires_grad=True)
f = (x**3).sum()
dfdx = torch.autograd.grad(f, x, create_graph=True)
print(dfdx)

print(dfdx[0][0])
dfdx[0][0].backward()
print(x.grad)
