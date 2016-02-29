# execute this test with "nosetests <file-name.py>"

import unittest
import numpy
import numpy as np
import numpy.testing as npt

from warpctc import ctc
from retry import retry

from chainer import cuda

def genLabel(alphabet_size, L):
    '''
    alphabet_size: int scalar
    L: int scalar
    return: int shape=(L)
    '''
    label = numpy.random.randint(1, alphabet_size-1, L)
    # guarantee repeats for testing
    if (L >= 3):
        label[L // 2] = label[L // 2 + 1]
        label[L // 2 - 1] = label[L // 2]
    return label

def genActs(size):
    '''
    size: int scalar
    return: float shape=(size)
    '''
    arr = numpy.random.uniform(0, 1, size)
    return arr

# Numerically stable softmax for a minibatch of 1
def softmax(acts):

    probs = np.zeros((acts.shape[0], acts.shape[2]))
    for t in range(0, acts.shape[0]):

        acts_sub = acts[t, 0, :]        

        max_activation = numpy.max(acts_sub)

        denom = np.sum(np.exp(acts_sub-max_activation))

        probs[t, :] = np.exp(acts_sub-max_activation)/denom

    return probs

# class TestSmall (unittest.TestCase):
#     alphabet_size = 5
#     T = 2
#     minibatch = 1
#     activations = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1]).reshape(
#                         T, minibatch, alphabet_size)
#     labels = [1, 2]
#
#     def test_small(self):
#         grads, score = ctc(self.activations, self.labels)
#         probs = softmax(self.activations)
#         expected_score = probs[0, 1] * probs[1, 2]
#         print(probs)
#         eps = 1e-6;
#
#         lb = expected_score - eps;
#         ub = expected_score + eps;
#
#         npt.assert_almost_equal(score, expected_score, decimal=6)
#         assert (score > lb and score < ub)

class TestWarpctc (unittest.TestCase):

    def test_small(self):
        alphabet_size = 5
        T = 2
        minibatch = 1
        activations = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1]).reshape(
                            T, minibatch, alphabet_size)

        print(activations.ravel())

        labels = [1, 2]

        grads, score = ctc(activations, labels)
        probs = softmax(activations)
        expected_score = probs[0, 1] * probs[1, 2]

        npt.assert_almost_equal(expected_score, 0.0851911)

        print(probs)

        npt.assert_almost_equal(numpy.exp(-score), expected_score, decimal=6)


    @retry(tries = 5)
    def test_inf(self):
        alphabet_size = 15
        T = 50
        L = 10
        minibatch = 1

        labels = genLabel(alphabet_size, L)
        activations = genActs(alphabet_size * T * minibatch).reshape(
                            T, minibatch, alphabet_size)
        activations[:, 0, 2] = -1e30

        grads, score = ctc(activations, labels)
        npt.assert_equal(score, float('Inf'))
        assert np.isnan(grads).any() == False

    def test_small_gpu(self):
        alphabet_size = 5
        T = 2
        minibatch = 1
        activations = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1]).reshape(
                            T, minibatch, alphabet_size)

        print(activations.ravel())

        activations_gpu = cuda.to_gpu(activations)

        labels = [1, 2]

        print(cuda.ndarray, type(activations_gpu))

        activations_gpu = cuda.cupy.asfortranarray(activations_gpu)
        grads, score = ctc(activations_gpu, labels, use_gpu=True)
        probs = softmax(activations)
        expected_score = probs[0, 1] * probs[1, 2]

        npt.assert_almost_equal(expected_score, 0.0851911)

        print(probs)

        npt.assert_almost_equal(numpy.exp(-score), expected_score, decimal=6)

    def test_grad_check(self):
        pass



















