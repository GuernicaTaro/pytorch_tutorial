import torch
from torch.autograd.functional import jacobian

def V(z):
  return (z**2).sum()

def x2z(x):
#  return x.exp()-10
  return 10*x.tanh()

x_init = torch.tensor([0.,1.,2.,3.,4.])
lr = 0.01
iter_max = 10000
tol = 1e-3

# initialization
iter = 1
x = x_init
x_list = [x_init.detach().numpy()]

while True:
  z = x2z(x)
  dzdx = jacobian(x2z, x)
  val = V(z)
  print(val)
  dVdz = jacobian(V, z)
  dVdx = torch.tensordot(dVdz, dzdx, dims=1)
  with torch.no_grad():
    x -= lr*dVdx
  x_list.append(x.detach().numpy())
  
  iter += 1
  if (iter >= iter_max) or (dVdx.norm() < tol):
    break
  
print('iter:      ',iter)
print('dvdx.norm: ', dVdx.norm())
print('x:         ',x)
