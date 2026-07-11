from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # accumulate the gradient: dL_i/dW[:, j] = (p[j] - 1{j == y[i]}) * X[i]
        for j in range(num_classes):
            indicator = 1.0 if j == y[i] else 0.0
            dW[:, j] += (p[j] - indicator) * X[i]


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Average over the batch and add the gradient of the L2 regularization.
    # The reg term in the loss is `reg * sum(W*W)` (no 0.5), so its gradient
    # is `2 * reg * W`.
    dW /= num_train
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*******
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    # Scores and shifted scores for numeric stability
    scores = X.dot(W)                                               # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)                 # (N, C)

    # Softmax probabilities
    exp_scores = np.exp(scores)                                     # (N, C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

    # Cross-entropy loss (averaged over batch) + L2 regularization
    correct_logprobs = -np.log(probs[np.arange(num_train), y])      # (N,)
    loss = np.sum(correct_logprobs) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*******
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Subtract 1 from the probability of the correct class for each example;
    # then dW = X^T @ (probs - one_hot(y)) / N.
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train

    dW = X.T.dot(dscores)                                           # (D, C)
    # Gradient of `reg * sum(W*W)` is `2 * reg * W`
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*******
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW
