from .layers import (
    affine_forward,
    affine_backward,
    relu_forward,
    relu_backward,
    batchnorm_forward,
    batchnorm_backward,
    layernorm_forward,
    layernorm_backward,
    dropout_forward,
    dropout_backward,
    conv_forward_naive,
    conv_backward_naive,
    max_pool_forward_naive,
    max_pool_backward_naive,
)


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer."""
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """Convenience layer: affine -> batchnorm -> relu."""
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """Backward pass for affine-batchnorm-relu."""
    fc_cache, bn_cache, relu_cache = cache
    da_bn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    """Convenience layer: affine -> layernorm -> relu."""
    a, fc_cache = affine_forward(x, w, b)
    a_ln, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    out, relu_cache = relu_forward(a_ln)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache


def affine_ln_relu_backward(dout, cache):
    """Backward pass for affine-layernorm-relu."""
    fc_cache, ln_cache, relu_cache = cache
    da_ln = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = layernorm_backward(da_ln, ln_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """Convenience layer: conv -> relu."""
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for conv-relu."""
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer: conv -> relu -> max-pool."""
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for conv-relu-pool."""
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db
