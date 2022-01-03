import torch
from torch.autograd.functional import jacobian

def my_exp(x):
  return x.exp()
inputs = torch.tensor([[4., 2.], [1., 5.]])
jac = jacobian(my_exp, inputs)
print(jac)
