# lorenz_4Dvar

4D var assimilation introduction using a chaotic lorenz model.

## lorenz model

The lorenz model consists in the evolution in time of 3 coordinates : x,y,z  
$$\frac{\partial x}{\partial t} = \sigma(x-y)$$
$$\frac{\partial y}{\partial t} = x(\rho - z) - y$$
$$\frac{\partial z}{\partial t} = xy - \beta z$$
Where $\sigma$,$\rho$ and $\beta$ are three reals.
