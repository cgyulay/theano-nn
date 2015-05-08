import theano
import cPickle
import gzip
import os
import time

import theano.tensor as T
import numpy as np
from theano import config

config.mode='FAST_COMPILE'


# Network

class ConnectedLayer():

  def __init__(self, input, n_in, n_out, rng, activation='relu', p_dropout=0.0, w=None, b=None):
    self.input = input
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.p_dropout = p_dropout

    if w is None:
      w = init_weights((n_in, n_out))

    if b is None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    self.output_no_dropout = activation(T.dot(input, self.w) + self.b)

    if p_dropout > 0:
      input = dropout(input, p_dropout, rng)

    self.output = activation(T.dot(input, self.w) + self.b)
    self.params = [self.w, self.b]


class LogisticRegressionLayer():

  def __init__(self, input, n_in, n_out, rng, activation='softmax', p_dropout=0.0, w=None, b=None):
    self.input = input
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.p_dropout = p_dropout

    if w is None:
      w = init_weights((n_in, n_out))

    if b is None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    self.output_no_dropout = activation(T.dot(input, self.w) + self.b)

    if p_dropout > 0:
      input = dropout(input, p_dropout, rng)

    self.output = activation(T.dot(input, self.w) + self.b)
    self.y_pred = T.argmax(self.output, axis=1)
    self.params = [self.w, self.b]


class NN():

  def __init__(self, lr=0.05, batch_size=100, n_hidden=100, n_epochs=500, dataset='mnist.pkl.gz'):
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.dataset = dataset

    # Load data and unpack
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print('Building the model...')

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)

    self.h1 = ConnectedLayer(
      input=x,
      n_in=28 * 28,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=0.5
    )

    self.h2 = ConnectedLayer(
      input=self.h1.output,
      n_in=n_hidden,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=0.5
    )

    self.softmax = LogisticRegressionLayer(
      input=self.h2.output,
      n_in=n_hidden,
      n_out=10,
      rng=rng,
      activation=softmax,
      p_dropout=0.5
    ) # softmax.output = py_x

    self.params = self.h1.params + self.h2.params + self.softmax.params

    dropped_py_x = self.softmax.output
    py_x = self.softmax.output_no_dropout
    y_pred = self.softmax.y_pred

    cost = ce_multiclass(dropped_py_x, y)
    updates = sgd(cost, self.params, self.lr)

    print('Compiling functions...')

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

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # valid_predictions = predict(valid_set_x)
    # valid_accuracy = np.mean(valid_predictions == valid_set_y)

    # test_predictions = predict(test_set_x)
    # test_accuracy = np.mean(test_predictions == test_set_y)

    print('Beginning training!')

    for epoch in xrange(n_epochs):
      start_time = time.clock()

      for start in xrange(n_train_batches):
        cost = train(start)

      test_predictions = predict(test_set_x.get_value(borrow=True))
      test_accuracy = np.mean(test_predictions == test_set_y.eval())

      end_time = time.clock()
      print('Epoch %d of %d took %.1fs\nTest accuracy: %.2f%%' % ((epoch + 1), n_epochs, (end_time - start_time), (test_accuracy * 100)))


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


