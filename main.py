import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from plots import *

   
# defining NN architecture
class HeatNet(nn.Module):
    def __init__(self):
        super(HeatNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 10),     # input: (x, y, t)
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)      # output: u(x, y, t)
        )

    def forward(self, x):
        return self.model(x)

# for collocation points we build a grid over the domain R3 with 0 <= x, y, t <= 1.
step = 0.1

# 1d axes
x = np.arange(0, 1, step)
y = np.arange(0, 1, step)
t = np.arange(0, 1, step)

# meshgrid creation
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# flattening into (N, 3) array of collocation points
collocation_points = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)

# converting to tensor, need gradient for later
X_f = torch.tensor(collocation_points, dtype=torch.float32, requires_grad=True)

# ------- loss function ----------
alpha = 0.05  # thermal diffusivity constant

# Lpde
def pde_loss(model, X_f, alpha=0.05):
    X_f.requires_grad_(True)
    u = model(X_f)

    # First-order gradients w.r.t. x, y, t
    grads = torch.autograd.grad(u, X_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    # Second-order gradients
    u_xx = torch.autograd.grad(u_x, X_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    residual = u_t - alpha * (u_xx + u_yy)

    return torch.mean(residual**2)

# Lic
def initial_condition_points(step=0.1):
    x = np.arange(0, 1, step)
    y = np.arange(0, 1, step)
    X, Y = np.meshgrid(x, y, indexing='ij')

    X0 = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X).ravel()], axis=1)
    X0 = torch.tensor(X0, dtype=torch.float32)

    u0 = torch.sin(np.pi * X0[:, 0]) * torch.sin(np.pi * X0[:, 1])
    u0 = u0.unsqueeze(1)  # shape [N, 1]

    #x y pairs and scalar temp
    return X0, u0

def ic_loss(model, X0, u0):
    u_pred = model(X0)
    return torch.mean((u_pred - u0)**2)

#Lbc

def boundary_condition_points(step=0.1):
    t = np.arange(0, 1, step)
    xy = np.arange(0, 1, step)

    boundary_pts = []

    for time in t:
        for xi in xy:
            # x = 0 and x = 1
            boundary_pts.append([0, xi, time])
            boundary_pts.append([1, xi, time])
            # y = 0 and y = 1
            boundary_pts.append([xi, 0, time])
            boundary_pts.append([xi, 1, time])

    X_b = torch.tensor(boundary_pts, dtype=torch.float32)
    return X_b

def bc_loss(model, X_b):
    u_pred = model(X_b)
    return torch.mean(u_pred**2)  # target is 0 everywhere

# final loss

def total_loss(model, X_f, X0, u0, X_b, alpha, l1=0.05, l2=0.01, l3=0.01):
    loss_pde = pde_loss(model, X_f, alpha)
    loss_ic = ic_loss(model, X0, u0)
    loss_bc = bc_loss(model, X_b)
    return l1*loss_pde + l2*loss_ic + l3*loss_bc, l1*loss_pde, l2*loss_ic, l3*loss_bc

# enerating training points
X_f = X_f.to('cpu')
X0, u0 = initial_condition_points(step=0.1)
X_b = boundary_condition_points(step=0.1)

X0 = X0.to('cpu')
u0 = u0.to('cpu')
X_b = X_b.to('cpu')

model = HeatNet().to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5000

loss_history = []
pde_history = []
ic_history = []
bc_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    loss, loss_pde, loss_ic, loss_bc = total_loss(model, X_f, X0, u0, X_b, alpha=0.05)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    pde_history.append(loss_pde.item())
    ic_history.append(loss_ic.item())
    bc_history.append(loss_bc.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# plot_loss_components(loss_history, pde_history, ic_history, bc_history)
# animate_real_2d(0.05, save_path=r"C:\Users\kalot\Desktop\real_2d.gif")
# animate_model_prediction(model, save_path=r"C:\Users\kalot\Desktop\model_2d.gif")
# animate_model_3d(model, save_path=r"C:\Users\kalot\Desktop\model_3d.gif")
# animate_real_3d(0.05, save_path=r"C:\Users\kalot\Desktop\real_3d.gif")
animate_error_3d(model, 0.05, save_path=r"C:\Users\kalot\Desktop\error_3d.gif")