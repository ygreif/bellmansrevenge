import tensorflow as tf
import numpy as np
import math

import neuralnetwork


class SetupNAF(object):

    @classmethod
    def setup(cls, env, nnvParameters, nnpParameters, nnqParameters, learningParameters):
        indim, actiondim = env.shape()
        x = tf.placeholder(tf.float32, [None, indim])
        nnv = neuralnetwork.NeuralNetwork(indim, 1,  x=x, **nnvParameters)
        nnq = neuralnetwork.NeuralNetwork(
            indim, actiondim, x=x, **nnqParameters)
        if actiondim == 1:
            pdim = 1
        else:
            pdim = (actiondim) * (actiondim + 1) / 2
        nnp = neuralnetwork.NeuralNetwork(
            indim, pdim, x=x, **nnpParameters)
        naf = NAFApproximation(
            nnv, nnp, nnq, actiondim, **learningParameters)
        return naf


def coldStart(naf, coldstart_len, n_examples):
    # initialize NAF so actions are in range
    n = 100
    success = False
    states = [env.sample_state() for _ in range(n)]
    max_prod = [(env.production.production(**state),) for state in states]
    if learningParameters['compress']:
        targets = [(0, ) for state in states]
    else:
        targets = [(state['k'] / 2, ) for state in states]
    states = [(state['k'], state['z']) for state in states]
    for i in range(coldstart_len):
        actions = naf.actions(states, max_prod)
        naf.train_actions_coldstart(states, max_prod, targets)
        if i % 10000 == 0:
            print "action", [a[0] for a in actions[0:10]]
            print "target", [t[0] for t in targets[0:10]]
        if np.allclose(actions, targets, atol=.2):
            success = True
            break
    if not success:
        print "WARNING actions did not converge"
        print naf.actions(states, max_prod)[0:5]
    success = False
    n = 100
    for i in range(coldstart_len):
        states = [env.sample_state() for _ in range(n)]
        targets = [(-10.0 + state['k'] * .01, ) for state in states]
        states = [(state['k'], state['z']) for state in states]
        naf.train_values_coldstart(states, targets)
        values = naf.value(states)
        if i % 10000 == 0:
            pass
            print "action", values[0:10]
            print "target", targets[0:10]
        if np.allclose(values, targets, atol=.2):
            success = True
            break
    if not success:
        print "WARNING values did not converge"
    return naf


class NAFApproximation(object):

    def to_semi_definite(self, M):
        diag = tf.sqrt(tf.exp(tf.matrix_diag_part(M)))
        L = tf.matrix_set_diag(M * self.mask, diag)
        return tf.matmul(L, tf.transpose(L))

    def __init__(self, nnv, nnp, nnq, actiondim, learning_rate, discount, compress, keep_prob=1):
        self.beta = discount
        self.discount = tf.constant(discount, dtype=tf.float32)
        self.x = nnv.x

        self._setup_v_calculation(nnv)
        self._setup_p_calculation(nnp, actiondim)
        self._setup_q_calculation(nnq, actiondim, compress)
        self._setup_next_q_calulcation()
        self._setup_train_step(learning_rate)
        self.keep_prob = keep_prob

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_v_calculation(self, nn):
        self.vx = nn.x
        self.v = nn.out

        # coldstart value
        self.target_value = tf.placeholder(
            tf.float32, [None, 1], name="target_value")
        coldstart_loss = tf.reduce_sum(
            tf.square(self.target_value - self.v))
        self.coldstart_values = tf.train.AdamOptimizer(
            learning_rate=.1).minimize(coldstart_loss)

    def _setup_p_calculation(self, nn, actiondim):
        mask = np.ones((actiondim, actiondim))
        mask[np.triu_indices(actiondim)] = 0
        self.mask = tf.constant(mask, dtype=tf.float32)
        self.px = nn.x
        upper_triang = tf.exp(
            tf.contrib.distributions.fill_triangular(nn.out))
        diag = tf.matrix_diag_part(upper_triang)
        L = tf.matrix_set_diag(upper_triang * mask, diag)
        self.P = tf.matmul(L, tf.transpose(L, perm=[0, 2, 1]))

    def _setup_q_calculation(self, nn, actiondim, compress=False):
        self.action_inp = tf.placeholder(
            tf.float32, [None, actiondim], name="action")
        self.max_prod = tf.placeholder(tf.float32, [None, 1], name="max_prod")

        self.qx = nn.x
        self.qout = nn.out
        if compress:
            self.mu = ((tf.tanh(nn.out) + 1.0) / 2.0) * self.max_prod
        else:
            print "Not compressed"
            self.mu = nn.out

        self.batch = tf.reshape(self.action_inp - self.mu, [-1, 1, actiondim])
        self.a = tf.reshape(tf.matmul(
            tf.matmul(self.batch, self.P), tf.transpose(self.batch, [0, 2, 1])), [-1, 1])
        self.Q = self.v - .5 * self.a

        # coldstart action
        self.target_action = tf.placeholder(
            tf.float32, [None, actiondim], name="target_action")
        coldstart_loss = tf.reduce_sum(
            tf.square(self.target_action - self.qout))
        self.coldstart_actions = tf.train.AdamOptimizer(
            learning_rate=.001).minimize(coldstart_loss)

    def _setup_next_q_calulcation(self):
        self.r = tf.placeholder(tf.float32, [None, 1], name="reward")

        self.update = self.v * self.discount + self.r

    def _setup_train_step(self, learning_rate):
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.actionloss = tf.reduce_sum(tf.abs(tf.to_float(tf.greater(self.mu, 0.8 * self.max_prod)) * self.mu + tf.to_float(tf.less(
            self.mu, .2 * self.max_prod)) * (.2 * self.max_prod - self.mu))) * 99999999
        self.loss = tf.reduce_sum(
            tf.square(self.target - self.Q))

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)

    def checkpoint(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.save(self.session, checkpoint_file)

    def restore(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_file)

    def calcA(self, x, action, max_prod):
        return self.session.run(self.a, feed_dict={self.x: x, self.max_prod: max_prod, self.action_inp: action, neuralnetwork.keep_prob: 1.0})

    def calcP(self, x):
        return self.session.run(self.P, feed_dict={self.x: x, neuralnetwork.keep_prob: 1.0})

    def value(self, x):
        return self.session.run(self.v, feed_dict={self.x: x, neuralnetwork.keep_prob: self.keep_prob})

    def actions(self, x, max_prod):
        return self.session.run(self.mu, feed_dict={self.x: x, self.max_prod: max_prod, neuralnetwork.keep_prob: 1.0})

    def action(self, x, max_prod):
        return self.session.run(
            self.mu, feed_dict={self.x: x, self.max_prod: max_prod, neuralnetwork.keep_prob: 1.0})[0]

    def calcActionloss(self, x, max_prod):
        return self.session.run(self.actionloss, feed_dict={self.x: x, self.max_prod: max_prod, neuralnetwork.keep_prob: 1.0})

    def calcq(self, rewards, next_state):
        return self.session.run(self.update, feed_dict={
            self.r: rewards, self.x: next_state, neuralnetwork.keep_prob: self.keep_prob})

    def storedq(self, state, action, max_prod):
        return self.session.run(self.Q, feed_dict={self.x: state, self.action_inp: action, self.max_prod: max_prod, neuralnetwork.keep_prob: 1.0})

    def calcloss(self, state, action, max_prod, rewards, next_state):
        target = self.calcq(rewards, next_state)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state, self.max_prod: max_prod, self.action_inp: action, neuralnetwork.keep_prob: self.keep_prob})

    def train_actions_coldstart(self, state, max_prod, target_action):
        return self.session.run(self.coldstart_actions, feed_dict={self.x: state, self.target_action: target_action, self.max_prod: max_prod, neuralnetwork.keep_prob: self.keep_prob})

    def train_values_coldstart(self, state, target_value):
        return self.session.run(self.coldstart_values, feed_dict={self.x: state, self.target_value: target_value, neuralnetwork.keep_prob: self.keep_prob})

    def trainstep(self, state, action, max_prod, rewards, next_state):
        target = self.calcq(rewards, next_state)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state, self.max_prod: max_prod, self.action_inp: action, neuralnetwork.keep_prob: self.keep_prob})

    def __exit__(self):
        self.session.close()

    def renderBestA(self, include_best=True):
        x = [(x, 0) for x in np.arange(0, 1.0, .01)]
        max_prod = [(math.pow(s[0], 1.0 / 3.0) * .9792,) for s in x]

        actions = self.actions(x, max_prod)
        best = [p[0] * (1 - 1.0 / 3.0 * .95) for p in max_prod]
        import matplotlib.pyplot as plt

        plt.plot([s[0] for s in x], actions, label="action")
        plt.plot([s[0] for s in x], max_prod, label="max")
        if include_best:
            plt.plot([s[0] for s in x], best, label="best")
        plt.title("Best Actions")
 #       plt.waitforbuttonpress(0)
        # plt.close()
        return plt

    def renderA(self):
        states = [(x, 0) for x in np.arange(.1, 1.1, .25)]
        #        max_prod = [(math.pow(x, 1.0 / 3.0) * .9792) for s in x]

        import matplotlib.pyplot as plt
        for state in states:
            mprod = math.pow(state[0], 1.0 / 3.0) * .9792
            actions = np.expand_dims(np.arange(0, mprod, mprod / 10.0), axis=1)
            max_prod = ([(mprod,) for _ in actions])

#            max_prod = [(math.pow(x, 1.0 / 3.0) * .9792,) for i in]
            value = self.calcA([state for a in actions], actions, max_prod)

            plt.plot(actions, value, label=str(state))
        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderQ(self):
        state = (.4, 0)
        mprod = math.pow(state[0], 1.0 / 3.0) * .9792

        actions = np.expand_dims(np.arange(.001, mprod, mprod / 10.0), axis=1)
        xx = [a[0] for a in actions]
        max_prod = [(mprod,) for _ in actions]
        utility = [(math.log(a[0]),) for a in actions]
        x = [state for _ in actions]
        next_x = [(mprod - a[0], 0) for a in actions]

        import matplotlib.pyplot as plt

        plt.plot(xx, self.storedq(x, actions, max_prod),
                 label="value + action reward")
        plt.plot(xx, -.5 * self.calcA(x, actions, max_prod), label="a reward")
        plt.plot(xx, self.calcq(utility, next_x),
                 label="utility + next_value")
        plt.plot(xx, self.value(next_x), label="next value")

        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderV(self):
        x = [(x, 0) for x in np.arange(0, 1.0, .01)]
        y = self.value(x)
        import matplotlib.pyplot as plt
        plt.plot([s[0] for s in x], y)
        plt.title("v over range")
        plt.waitforbuttonpress(0)
        plt.close()
