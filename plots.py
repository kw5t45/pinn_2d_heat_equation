import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch


# exact - ground truth solution
def u_exact(x, y, t, alpha):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi ** 2 * alpha * t)


def animate_real_2d(alpha, save_path=None):
    """
    Plots the heat distribution evolution on a 2d graph.
    :param save_path:
    :param alpha: thermal diffusivity constant
    :return:
    """
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    Z = u_exact(X, Y, 0, alpha)
    contour = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='hot', vmin=0, vmax=1)
    fig.colorbar(contour)
    ax.set_title("Heat Equation Analytical Solution")

    def update(frame):
        t = frame / 100
        Z = u_exact(X, Y, t, alpha)
        contour.set_data(Z)
        ax.set_title(f"Heat Equation Analytical Solution for t = {t:.2f}")
        return [contour]

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=101, interval=100, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

def animate_real_3d(alpha, save_path=None):
    """
    Plots the heat distribution evolution on a 3d graph.
    :param save_path:
    :param alpha: thermal diffusivity constant
    :return:
    """
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 1)

    #  surface
    Z = u_exact(X, Y, 0, alpha)
    surf = ax.plot_surface(X, Y, Z, cmap='inferno')

    def update(frame):
        ax.clear()
        t = frame / 100
        Z = u_exact(X, Y, t, alpha)
        surf = ax.plot_surface(X, Y, Z, cmap='inferno')
        ax.set_zlim(0, 1)
        ax.set_title(f"Heat Equation at t = {t:.2f}")
        return [surf]

    #  animation
    ani = animation.FuncAnimation(fig, update, frames=101, interval=100, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

def animate_model_prediction(model, device='cpu', save_path=None):
    """
    Animates the model-predicted heat distribution over time using a 2D heatmap.
    :param save_path:
    :param model: Trained PINN model
    :param device: 'cpu' or 'cuda'
    """
    model.eval()

    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    Z = np.zeros_like(X)
    contour = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='hot', vmin=0, vmax=1)
    fig.colorbar(contour)
    ax.set_title("Model Prediction")

    def update(frame):
        t = frame / 100
        T = np.full_like(X, t)
        input_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(device)

        with torch.no_grad():
            u_pred = model(input_tensor).cpu().numpy().reshape(X.shape)

        contour.set_data(u_pred)
        ax.set_title(f"Model Prediction at t = {t:.2f}")
        return [contour]

    ani = animation.FuncAnimation(fig, update, frames=101, interval=100, blit=False)
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

def animate_model_3d(model, device='cpu', save_path=None):
    """
    Animates the model-predicted heat distribution over time in 3D.
    :param save_path:
    :param model: Trained PINN model
    :param device: 'cpu' or 'cuda'
    """
    model.eval()

    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 1)

    # Initial prediction
    T0 = np.full_like(X, 0)
    input_grid = np.stack([X.ravel(), Y.ravel(), T0.ravel()], axis=1)
    input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        Z = model(input_tensor).cpu().numpy().reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap='inferno')

    def update(frame):
        ax.clear()
        t = frame / 100
        T = np.full_like(X, t)
        input_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(device)

        with torch.no_grad():
            Z = model(input_tensor).cpu().numpy().reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap='inferno')
        ax.set_zlim(0, 1)
        ax.set_title(f"Model Prediction at t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x, y, t)")
        return [surf]

    ani = animation.FuncAnimation(fig, update, frames=101, interval=100, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()


def plot_loss_history(loss_history, total_epochs):
    """
    Plots the training loss over epochs.

    :param loss_history: List of loss values recorded during training
    :param total_epochs: Total number of training epochs
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss over {total_epochs} Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_components(loss_history, pde_history, ic_history, bc_history):
    """
    Plots total loss and individual loss components over epochs.
    """
    plt.figure(figsize=(10, 5))
    epochs = range(len(loss_history))

    plt.plot(epochs, loss_history, label='Total Loss', color='black', linewidth=2)
    plt.plot(epochs, pde_history, label='PDE Loss', linestyle='--')
    plt.plot(epochs, ic_history, label='IC Loss', linestyle='--')
    plt.plot(epochs, bc_history, label='BC Loss', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss and Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def animate_error_3d(model, alpha, device='cpu', save_path=None):
    """
    Animates the squared error ||u_pred - u_real||^2 over time in 3D.
    :param model: Trained PINN model
    :param alpha: thermal diffusivity constant
    :param device: 'cpu' or 'cuda'
    :param save_path: optional path to save the animation as GIF
    """
    model.eval()

    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 0.1)  # Adjust based on typical error scale

    def update(frame):
        ax.clear()
        t = frame / 100
        T = np.full_like(X, t)

        input_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(device)

        with torch.no_grad():
            u_pred = model(input_tensor).cpu().numpy().reshape(X.shape)

        u_real = u_exact(X, Y, t, alpha)
        error = (u_pred - u_real) ** 2

        surf = ax.plot_surface(X, Y, error, cmap='coolwarm')
        ax.set_zlim(0, np.max(error) * 1.2)
        ax.set_title(f"Squared Error ||u_pred - u_real||² at t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Error")
        return [surf]

    ani = animation.FuncAnimation(fig, update, frames=101, interval=100, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
