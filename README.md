# lorenz_4Dvar

This repository contains files to launch a 4Dvar  assimilation using a chaotic lorenz model.

## lorenz system

The lorenz system is well known for its chaotic behavior. It is described by three coordinates (x,y,z) which evolutions in time is controlled by the following equations. 
$$\frac{\partial x}{\partial t} = \sigma(x-y)$$
$$\frac{\partial y}{\partial t} = x(\rho - z) - y$$
$$\frac{\partial z}{\partial t} = xy - \beta z$$
Where $\sigma$,$\rho$ and $\beta$ are three reals.

## Data assimilation

Data assimilation is a branch of mathematic which consist in combining a theoretical model with observations. The model usually describes the evolution in time of a physical phenomenon for which a bunch of observations are available. Knowledges from both the model and the observations are then combined using data assimilation methods to find for a specific goal. It can be the identification of the initial conditions in order for the model to best fit the observations, to interpolate sparse observations of a system, to determine the optimal state estimate of a system.
Variational assimilation methods where first developped with applications to weather forecast. The weather system being highly chaotic, no model can actually predict its evolution after a certain amount of time. A few days to weeks when the weather is stable. The assimilation of live observations in a model prevent it to diverge from reality. 

## 4Dvar

The 4Dvar is a four-dimensional variational assimilation method. It considered that both observations and prior knowledge on the system (initial conditions) are characterized by errors which follow normal laws.
The state of the system is described by the state vector $X$. The 4Dvar algorithm find the optimal state of the system that best fit both observations and previous kbowledge of the system by minimizing the cost function $J$ :

$$J(X)=(X-X_{b})^{T}B^{-1}(X-X_{b})+\sum_{i}(y_{i}-H_{i}(x_{i}))^{T}R_{i}^{-1}(y_{i}-H_{i}(x_{i}))$$

where $X_{b}$ is the background state. It corresponds to a prior knwoledge of the system. The $y_{i}$ are the observations and the $x_{i}$, the state of the model with for initial conditions $X$, at the time where the observation $y_{i}$ is available. $H_{i}$ is called the observation operator, it project the state of the model in the same space as the observations.
