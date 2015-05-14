import theano
import time

import theano.tensor as T
import numpy as np
from theano import config

import class_mnist_nn as nn

config.mode='FAST_COMPILE'
config.floatX='float32'


class SoftmaxRegression():

  def __init__(self, lr=0.5, batch_size=200, n_epochs=100, dataset='mnist.pkl.gz', L2_reg=0.0001):
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.dataset = dataset

    # Load data and unpack
    datasets = nn.load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Number of minibatches
    n_train_batches = len(train_set_x.get_value(borrow=True)) / batch_size
    n_valid_batches = len(valid_set_x.get_value(borrow=True)) / batch_size
    n_test_batches = len(test_set_x.get_value(borrow=True)) / batch_size

    # Number of target classes
    classes = set()
    for item in train_set_y.eval():
      classes.add(item)
    n_output_classes = len(classes)

    print('Building the model...')

    x = T.matrix('x')
    y = T.ivector('y')

    # Multiclass classifier through softmax
    softmax = nn.SoftmaxLayer(
      input=x,
      n_in=28 * 28,
      n_out=n_output_classes
    )

    self.params = softmax.params

    # Regularization
    L2 = (softmax.w ** 2).sum()

    # Construct model
    py_x = softmax.output # softmax output = P(y|x)

    y_pred = softmax.y_pred
    accuracy = T.mean(T.eq(y_pred, y))
    cost = nn.ce_multiclass(py_x + L2 * L2_reg, y)
    updates = nn.sgd(cost, self.params, self.lr)

    print('Compiling functions...')

    # Use givens to specify numeric minibatch from symbolic x and y
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

    predict = theano.function(
      inputs=[x],
      outputs=y_pred
    )

    print('Beginning training!')

    for epoch in xrange(n_epochs):
      start_time = time.clock()

      for start in xrange(n_train_batches):
        cost = train(start)

      # total_valid_accuracy = np.mean([valid_accuracy(i) for i in xrange(n_valid_batches)])
      total_test_accuracy = np.mean([test_accuracy(i) for i in xrange(n_test_batches)])

      end_time = time.clock()
      print('\nEpoch %d of %d took %.1fs\nTest accuracy: %.2f%%' % ((epoch + 1), n_epochs, (end_time - start_time), (total_test_accuracy * 100)))



# Run

if __name__ == '__main__':
  SoftmaxRegression(lr=0.1)

