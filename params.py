import random

from envs import economy
import numpy as np

import torch.nn as nn

vProductivity = np.array([0.9792], dtype=float)
mTransition = np.array([[1.0]], dtype=float)

alpha = 1.0 / 3.0
delta = 1.0
prod = economy.Production(alpha, delta, vProductivity)
motion = economy.Motion(mTransition)
model = economy.GrowthEconomy(economy.LogUtility(), prod, motion)


def random_nn_parameters(hidden_choices=[[5], [10], [100], [200], [200, 200], [400], [50], [10, 10], [10, 10, 10], [100, 100], [100, 100, 100], [1000]], nonlinearity_cls_choices=[nn.LeakyReLU]):
    return {'hidden_layers': random.choice(hidden_choices), 'nonlinearity_cls': np.random.choice(nonlinearity_cls_choices)}


def random_learning_parameters(rates=[.1, .01, .001, .002, .0001], discount=[.95], compress=[True]):
    return {'learning_rate': np.random.choice(rates), 'discount': np.random.choice(discount), 'compress': np.random.choice(compress)}


def random_strat(lo=[.5, .75, .8, .9], hi=[1.1, 1.2, 1.5, 1.25], max_iters=[100], minibatch_size=[50, 100, 500, 1000]):
    return {'lo': np.random.choice(lo), 'hi': np.random.choice(hi), 'max_iters': np.random.choice(max_iters), 'minibatch_size': np.random.choice(minibatch_size)}


def random_delta_strat(delta=[.001, .01, .05, .1], max_iters=[100, 200], minibatch_size=[50, 100, 500, 1000]):
    return {'delta': np.random.choice(delta), 'max_iters': np.random.choice(max_iters), 'minibatch_size': np.random.choice(minibatch_size)}


def random_parameters():
    nnvParameters = random_nn_parameters()
    nnpchoices = [[], [5], [10], [10], [5, 5], [25], [25, 25]]
    nnpParameters = random_nn_parameters(
        hidden_choices=nnpchoices)
    nnqParameters = random_nn_parameters()
    learningParameters = random_learning_parameters()
    strat = random_strat()
    parameters = {'naf': {
        'nnvParameters': nnvParameters, 'nnpParameters': nnpParameters,
        'nnqParameters': nnqParameters, 'learningParameters': learningParameters}, 'strat': strat}
    return parameters
