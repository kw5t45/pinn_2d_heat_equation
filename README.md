# pinn_2d_heat_equation
### Physics Informed Neural Network predicting temperature on a surface with Dirichlet boundary conditions.
PINNs are neural networks trained not only on data but also to satisfy **underlying physical laws**, represented as differential equations. Here, we solve a PDE **without labeled training data** — only the governing equation and boundary/initial conditions are used during training.

![test](resources/model_3d.gif)
## Equation and model description
For values of $0 <= x, y, z <= 1$ we know that the heat equation is 
<pre>```math\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
```</pre>
Given the boundary conditions 