from jax.nn import softmax
from jax import random
from jax.random import uniform, gumbel
import jax.numpy as np
from jax import grad, jit, vmap
from einops import repeat

def weights_to_probs(weights):
    return softmax(weights)

n = 4
key = random.PRNGKey(0)
key, subkey = random.split(key)
weights = uniform(key, (n,))
probs = weights_to_probs(weights)

assert np.isclose(np.sum(probs), 1)

print("(*) First Model")

def sampler_model(weights, key):
    probs = weights_to_probs(weights)
    logprobs = np.log(probs)
    noise = gumbel(key, shape=(n,))
    perturbed = logprobs + noise
    smooth = softmax(perturbed)
    return smooth

samples = 100
key = random.PRNGKey(0)
key, *subkeys = random.split(key, num=samples + 1)
subkeys = np.array(subkeys)

smooths = vmap(sampler_model)(repeat(weights, "p -> b p", b=samples), subkeys)
argmaxes = np.argmax(smooths, axis=1)
freqs = np.array([len([e for e in argmaxes if e == f]) / samples for f in range(n)])

def opt_target_distrib(weights, target, key):
    key, *subkeys = random.split(key, num=samples + 1)
    subkeys = np.array(subkeys)
    batch_weights = repeat(weights, "p -> b p", b=samples)
    smooths = vmap(sampler_model)(batch_weights, subkeys)
    aggregate = np.mean(smooths, axis=0)
    return np.mean(np.power(aggregate - target, 2))

key = random.PRNGKey(0)
key, subkey = random.split(key)
target_weights = uniform(subkey, (n,))
target_probs = weights_to_probs(target_weights)
key, subkey = random.split(key)

epochs = 1
g = grad(opt_target_distrib)
for epoch in range(epochs):
    key, subkey = random.split(key)
    weights += -g(weights, target_probs, subkey)
    print(weights_to_probs(weights))

print(target_probs, "(target)")

def opt_simplex(weights, target, key):
    key, *subkeys = random.split(key, num=samples + 1)
    subkeys = np.array(subkeys)
    batch_weights = repeat(weights, "p -> b p", b=samples)
    smooths = vmap(sampler_model)(batch_weights, subkeys)
    norms = np.linalg.norm(smooths, axis=0)
    return np.mean(norms)

print("distrib norm optimization")

epochs = 100
g = grad(opt_simplex)
for epoch in range(epochs):
    key, subkey = random.split(key)
    weights += g(weights, target_probs, subkey)
    print(weights_to_probs(weights))
