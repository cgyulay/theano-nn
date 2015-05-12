import theano
import cPickle
import gzip
import os
import time

import theano.tensor as T
import numpy as np
from theano import config

config.mode='FAST_COMPILE'


# Activations

def linear(x):
  return x

def sigmoid(x):
  # T.nnet.sigmoid
  return 1.0 / (1.0 + T.exp(-x))

def tanh(x):
  return (T.exp(x) - T.exp(-x)) / (T.exp(x) + T.exp(-x))

def relu(x):
  return T.maximum(x, 0.0)

def relu_leaky(x, alpha=3.0):
  return T.maximum(x, x * (1.0 / alpha))


# Costs

def ce_multiclass(py_x, y):
  # Locations 1,2,3,...,n by y
  # T.nnet.categorical_crossentropy
  return -T.mean(T.log(py_x)[T.arange(y.shape[0]), y])

def ce_binary(py_x, y):
  # T.nnet.binary_crossentropy
  return T.mean(T.nnet.binary_crossentropy(py_x, y))

def mean_squared_error(pred, y):
  return T.mean((pred - y) ** 2)


# Network

class ConnectedLayer():

  def __init__(self, input, n_in, n_out, rng, activation=relu, p_dropout=0.0, w=None, b=None, input_no_dropout=None):
    self.input = input
    self.input_no_dropout = input_no_dropout
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.p_dropout = p_dropout

    if w == None:
      w = init_weights((n_in, n_out))

    if b == None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    self.output_no_dropout = activation(T.dot(input_no_dropout, self.w) + self.b)

    self.output = activation(T.dot(dropout(self.input, p_dropout, rng), self.w) + self.b)
    self.output_no_dropout = activation(T.dot(self.input, self.w) + self.b)
    self.params = [self.w, self.b]


class SoftmaxLayer():

  def __init__(self, input, n_in, n_out, rng, p_dropout=0.0, w=None, b=None, input_no_dropout=None):
    self.input = input
    self.input_no_dropout = input_no_dropout
    self.n_in = n_in
    self.n_out = n_out
    self.p_dropout = p_dropout

    if w == None:
      w = init_weights((n_in, n_out))

    if b == None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    # self.output = a if uses_dropout else b
    self.output = softmax(T.dot(dropout(self.input, p_dropout, rng), self.w) + self.b)
    self.output_no_dropout = softmax(T.dot(self.input, self.w) + self.b)
    self.y_pred = T.argmax(self.output, axis=1)
    self.params = [self.w, self.b]


class NN():

  def __init__(self, lr=0.1, batch_size=100, n_hidden=200, n_epochs=100, dataset='mnist.pkl.gz'):
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.dataset = dataset

    # Load data and unpack
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    classes = set()
    for item in train_set_y.eval():
      classes.add(item)
    n_output_classes = len(classes)

    print('Building the model...')

    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)

    self.h1 = ConnectedLayer(
      input=x,
      input_no_dropout=x,
      n_in=28 * 28,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=0.2
    )

    self.h2 = ConnectedLayer(
      input=self.h1.output,
      input_no_dropout=self.h1.output_no_dropout,
      n_in=n_hidden,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=0.5
    )

    self.softmax = SoftmaxLayer(
      input=self.h2.output,
      input_no_dropout=self.h2.output_no_dropout,
      n_in=n_hidden,
      n_out=n_output_classes,
      rng=rng,
      p_dropout=0.5
    ) # softmax.output = py_x

    # It looks ridiculous but just flattens all layer params into one list
    self.layers = [self.h1, self.h2, self.softmax]
    params = [self.layers[i].params for i in xrange(len(self.layers))]
    self.params = [param for subparams in params for param in subparams]

    dropped_py_x = self.softmax.output_no_dropout
    py_x = self.softmax.output_no_dropout
    y_pred = self.softmax.y_pred
    accuracy = T.mean(T.eq(y_pred, y))

    cost = ce_multiclass(dropped_py_x, y)
    updates = sgd(cost, self.params, self.lr)

    print('Compiling functions...')

    index = T.lscalar()
    train = theano.function(
      inputs=[index],
      outputs=cost,
      updates=updates,
      givens={
        x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
        y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
      }
    )

    predict = theano.function(
      inputs=[x],
      outputs=y_pred
    )

    valid_accuracy = theano.function(
      inputs=[index],
      outputs=accuracy,
      givens={
        x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
        y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
      }
    )

    test_accuracy = theano.function(
      inputs=[index],
      outputs=accuracy,
      givens={
        x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
        y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
      }
    )

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print('Beginning training!')

    for epoch in xrange(n_epochs):
      start_time = time.clock()

      for start in xrange(n_train_batches):
        cost = train(start)

      # total_valid_accuracy = np.mean([valid_accuracy(i) for i in xrange(n_valid_batches)])
      total_test_accuracy = np.mean([test_accuracy(i) for i in xrange(n_test_batches)])

      end_time = time.clock()
      print('\nEpoch %d of %d took %.1fs\nTest accuracy: %.2f%%' % ((epoch + 1), n_epochs, (end_time - start_time), (total_test_accuracy * 100)))


# Loading data

def load_data(dataset):
  print('Loading the data...')

  # load gzip from data folder with name dataset
  dataset = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()
  # train_set, valid_set, test_set format: tuple(input, target)
  # input is a 2d (matrix) numpy.ndarray whose rows correspond to an example
  # target is a 1d (vector) numpy.ndarray that has the same length as the number of rows in the input

  def shared_dataset(data_xy, borrow=True):
    # store dataset in theano shared variables to utilize gpu
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    return shared_x, T.cast(shared_y, 'int32')

  test_set_x, test_set_y = shared_dataset(test_set)
  valid_set_x, valid_set_y = shared_dataset(valid_set)
  train_set_x, train_set_y = shared_dataset(train_set)

  return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
          (test_set_x, test_set_y)]


# Utilities

def floatX(x):
  return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
  return theano.shared(floatX(np.random.randn(*shape) * 0.01))


# Gradient descent

def sgd(cost, params, lr=0.05):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    updates.append([p, p - g * lr])
  return updates


# Layers

def softmax(x):
  # T.nnet.softmax
  e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x')) # Make column out of 1d vector
  return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(x, p=0.0, rng=np.random.RandomState(1234)):
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

  if p > 0:
    p = 1 - p # 1 - p because p = prob of dropping
    x *= srng.binomial(x.shape, p=p, dtype=theano.config.floatX)
    x /= p
  return x


# Run

if __name__ == '__main__':
  NN()
