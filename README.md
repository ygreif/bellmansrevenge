# Bellmans Revenge

## Overview

This project explores using deep learning to solve [Dynamic Stochastic General Equilibrium Models](https://www.theigc.org/sites/default/files/2015/09/ISI2_2015.pdf) (DSGE), which have traditionally been solved using Chebyshev polynomials as function approximates.

I find that *value iteration using neural networks works extremely well for DSGE models*. Key insights include that

* *Exploration of the action space is less critical* than in traditional RL tasks. Policy functions in DSGE models tend to be smooth, so strategies learned off the equilibrium path often generalize well to it.
* *Cold-starting the actor and critic networks with reasonable priors is crucial*. DSGE value functions are often flat, increasing the risk of convergence to poor local optima without informed initialization. Economic theory can often supply these priors.
   - DDPG only succeeds if the actor and critics are coldstarted
   - Once the state space is larger than two parameters policy iteraion using a neural grid also needs coldstarting to converge well

## Highlights and Design Choices

* I tensorize the q-function and transition dynamics, to dramatically speed up the neural grid
* I use economic theory to guide model training. Including looking at the
 - The [https://projects.iq.harvard.edu/files/econ2010c/files/lecture_03_2010c_2014.pdf](Euler loss)
 - The deviation between the optimal action and the chosen action when delta=1 and there is a closed form solution

# Writeup

See [Deep Q Learning in Macroeconomics](Deep Q Learning in Macroeconomics.ipynb) for a more complete writeup

# Environments

The DSGE models are implemented in envs/. The core model is based on A Comparison of Programming Languages in Economics. Two variants are included:

- jesusfv. The standard RBC model where only capital and technology vary
- jesusfv_extended. An extension where the depreciation rate (delta) also varies

See [A Comparison of Programming Languages in Economics](https://github.com/jesusfv/Comparison-Programming-Languages-Economics) for more context.

# Code

* python qtable/grid.py              # policy iteration using a fixed grid
* python train_neural_grid.py        # policy iteration using a neural network function approximator
* python train_rl.py                 # Actor-Critic (DDPG) training

# Sanity Checks

* python test_euler_error.py         # tests that the Euler Error is low on the equilbrium path
* python test_neural_grid.py         # confirms that the parallelized tensor based Q function matches the iterative Q function
* python envs/tensor_utils.py        # confirms that the parallelized tensor based state transitions match the iterative ones
