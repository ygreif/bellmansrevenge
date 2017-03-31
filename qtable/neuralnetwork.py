import tensorflow as tf

keep_prob = tf.Variable(1.0)


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False, use_dropout=False):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))

        if nonlinearity:
            h = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            h = tf.matmul(inp, self.W) + self.b
        if use_dropout:
            self.out = tf.nn.dropout(h, keep_prob)
        else:
            self.out = h


class NeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers, nonlinearity=tf.nn.tanh, use_dropout=False, x=None):
        self.layers = []
        if x is not None:
            self.x = x
        else:
            print "Making new placeholder"
            self.x = tf.placeholder(tf.float32, [None, indim])
        self.indim = indim
        self.enddim = enddim

        inp = self.x
        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(inp, (prev_dim, out_dim), nonlinearity=nonlinearity, use_dropout=use_dropout))
            inp = self.layers[-1].out
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(
            inp, (prev_dim, enddim), nonlinearity=False))
        self.out = self.layers[-1].out
