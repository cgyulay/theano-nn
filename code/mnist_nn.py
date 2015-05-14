import theano
import cPickle
import gzip
import os
import time

import theano.tensor as T
import numpy as np
from theano import config

config.mode='FAST_COMPILE'

def floatX(X):
  return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
  return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def logistic(X):
  return 1.0 / (1.0 + T.exp(-X))

def relu(X):
  return T.maximum(X, 0.0)

def leaky_relu(X, alpha=3.0):
  return T.maximum(X, X * (1.0 / alpha))

def softmax(X):
  e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x')) # make column out of 1d vector
  return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.0, rng=np.random.RandomState(1234)):
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

  if p > 0:
    p = 1 - p # 1 - p because p = prob of dropping
    X *= srng.binomial(X.shape, p=p, dtype=theano.config.floatX)
    X /= p
  return X

def model(X, w_h, b_h, w_h2, b_h2, w_soft, b_soft, p_drop_input, p_drop_hidden, rng):
  X = dropout(X, p_drop_input, rng)
  h = relu(T.dot(X, w_h) + b_h)

  h = dropout(h, p_drop_hidden, rng)
  h2 = relu(T.dot(h, w_h2) + b_h2)

  h2 = dropout(h2, p_drop_hidden, rng)
  py_x = softmax(T.dot(h2, w_soft) + b_soft)
  # py_x = T.nnet.softmax(T.dot(h2, w_soft))

  return h, h2, py_x

# doesn't work atm
def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    acc = theano.shared(p.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * g ** 2
    gradient_scaling = T.sqrt(acc_new + epsilon)
    g = g / gradient_scaling
    updates.append((acc, acc_new))
    updates.append((p, p - lr * g))
  return updates


def sgd(cost, params, lr=0.05):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    updates.append([p, p - g * lr])
  return updates

def cross_entropy_cost(py_x, y):
  # locations 1,2,3,...,n by y
  return -T.mean(T.log(py_x)[T.arange(y.shape[0]), y])


# load gzip from data folder with name dataset
def load_data(dataset):
  print('Loading the data...')

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


def main(lr=0.05, batch_size=100, n_epochs=1000, dataset='mnist.pkl.gz'):

  # load data and unpack
  datasets = load_data(dataset)
  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  print('Building the model...')

  index = T.lscalar()
  X = T.matrix('X')
  Y = T.ivector('Y')
  rng = np.random.RandomState(1234)

  # 28 * 28 = 784
  # input <batch_size, 784>
  # h1 relu <784, 100>
  # h2 relu <100, 100>
  # softmax <100, 10>

  w_h = init_weights((784, 100))
  b_h = init_weights((100,))
  w_h2 = init_weights((100, 100))
  b_h2 = init_weights((100,))
  w_soft = init_weights((100, 10))
  b_soft = init_weights((10,))

  # todo: add biases b_h, b_h2, b_soft

  # model with dropout of 0.2 in input units and 0.5 in hidden units
  dropped_h, dropped_h2, dropped_py_x = model(X, w_h, b_h, w_h2, b_h2, w_soft, b_soft, 0.2, 0.5, rng)

  # model without dropout for prediction
  h, h2, py_x = model(X, w_h, b_h, w_h2, b_h2, w_soft, b_soft, 0.0, 0.0, rng)

  y_x = T.argmax(py_x, axis=1)

  # cost = T.mean(T.nnet.categorical_crossentropy(dropped_py_x, Y))
  cost = cross_entropy_cost(dropped_py_x, Y)
  params = [w_h, b_h, w_h2, b_h2, w_soft, b_soft]
  updates = sgd(cost, params, lr)

  print('Compiling functions...')

  train = theano.function([X, Y], cost, updates=updates, allow_input_downcast=True)
  predict = theano.function([X], y_x, allow_input_downcast=True)

  # from logistic_sgd
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

  # normally using a shared variable in a theano function utilizes it both as a symbolic and
  # numeric value. givens allows us to replace with a specific numeric value (each batch)
  train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
      X: train_set_x[index * batch_size: (index + 1) * batch_size],
      Y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
  )

  print('Beginning training!')

  length = train_set_x.get_value(borrow=True).shape[0]
  for i in range(n_epochs):
    start_time = time.clock()

    # for start in range(0, length, batch_size): # start, stop, step
    for start in xrange(n_train_batches):
      # x = train_set_x.get_value(borrow=True)[start: start + batch_size]
      # y = train_set_y.eval()[start: start + batch_size]

      # cost = train(x, y) # train mini-batch
      cost = train_model(start)

    # valid_predictions = predict(valid_set_x.get_value(borrow=True))
    # valid_accuracy = np.mean(valid_predictions == valid_set_y.eval())

    test_predictions = predict(test_set_x.get_value(borrow=True))
    test_accuracy = np.mean(test_predictions == test_set_y.eval())

    end_time = time.clock()
    print('Epoch %d of %d took %.1fs\nTest accuracy: %.2f%%' % ((i + 1), n_epochs, (end_time - start_time), (test_accuracy * 100)))

if __name__ == '__main__':
  lr = 0.05
  batch_size = 100
  n_epochs = 1000

  main(lr=lr, batch_size=batch_size, n_epochs=n_epochs)
