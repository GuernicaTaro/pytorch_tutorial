import torch

x = torch.ones(2, 3, requires_grad = True)
y = x + 2
z = y * y * 3
print(z) # z is tensor 3x3, has atttribute grad_fn

out = z.mean()
print(out) # out is scalar, has attribute grad_fn

print(x.grad) # x.grad not have value
out.backward() # backpropagation, x.grad <- x.grad + dout/dx
print(x.grad) # x.grad has value
print(out)

z2 = x * x * 3
out2 = z2.mean()
out2.backward() # increment of x.grad by dout2/dx
print(x.grad)
