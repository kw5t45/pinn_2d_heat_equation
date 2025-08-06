# pinn_2d_heat_equation
$`\sqrt{3x-1}+(1+x)^2`$
### Physics Informed Neural Network predicting temperature on a surface with Dirichlet boundary conditions.
PINNs are neural networks trained not only on data but also to satisfy **underlying physical laws**, represented as differential equations. Here, we solve a PDE **without labeled training data** — only the governing equation and boundary/initial conditions are used during training.

<img src="resources/model_3d.gif" alt="model" width="400"/>

## Equation and model description
For values of $0 <= x, y, z <= 1$ we know that the heat equation is <br><br>
$` \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) `$
<br><br>
Given the boundary condition <br><br>
$`u(x, y, t) = 0 \quad \text{for} \quad (x, y) \in \partial \Omega, \quad t > 0$<br><br>
and the initial condition<br><br>
$u(x, y, 0) = \sin(\pi x) \sin(\pi y) \quad \text{for} \quad (x, y) \in \Omega$<br><br>
we train a PINN to predict the scalar output $u(x, y, t)$ for a $0 <= x ,y ,z <= 1$ pair input using the analytic solution of the PDE and the conditions as our loss function.
## Model architecture
We train a simple fully connected MPL with 3 input channels, 2 hidden layers of 10 nodes each and the single output which represents the scalar value - temperature at given coordinates. Tanh activations are used in the hidden and output layers. 
<br>
<img src="resources/pinn_arch.png" alt="pinn" width="500"/>
