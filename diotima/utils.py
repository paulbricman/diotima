import jax.numpy as np


def norm(x, axis=-1, keepdims=False, eps=1e-7):
    return np.sqrt((x * x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x, axis=-1, eps=1e-20):
    return x / norm(x, axis, keepdims=True, eps=eps)
