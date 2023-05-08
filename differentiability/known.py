from jax.nn import softmax
from jax import random
from jax.random import uniform
import jax.numpy as np
from jax import grad, jit

# Building Blocks

def weights_to_probs(weights):
    return softmax(weights)

n = 4
key = random.PRNGKey(0)
key, subkey = random.split(key)
weights = uniform(key, (n,))
probs = weights_to_probs(weights)

assert np.isclose(np.sum(probs), 1)

def entropy(probs):
    logs = np.log2(probs)
    logs = np.nan_to_num(logs)
    infos = np.multiply(probs, logs)
    return -np.sum(infos)

min_entropy_probs = np.zeros(n).at[0].set(1)
max_entropy_probs = weights_to_probs(np.ones(n))

assert entropy(min_entropy_probs) == 0
assert entropy(max_entropy_probs) == np.log2(n)
assert entropy(probs) > entropy(min_entropy_probs)
assert entropy(probs) < entropy(max_entropy_probs)

# Maximizing and Minimizing Entropy

epochs = int(1e4)
print("(*) First Model (Minimizing/maximizing entropy)")

def first_model(weights):
    probs = weights_to_probs(weights)
    return entropy(probs)

g = jit(grad(first_model))
for epoch in range(epochs):
    weights += -g(weights)

print(weights_to_probs(weights), first_model(weights), "(target: 0)")
assert np.isclose(first_model(weights), entropy(min_entropy_probs), atol=1e-3)

key, subkey = random.split(key)
weights = uniform(key, (n,))

for epoch in range(epochs):
    weights += g(weights)

print(weights_to_probs(weights), first_model(weights), "(target: " + str(np.log2(n)) + ")")
assert np.isclose(first_model(weights), entropy(max_entropy_probs), atol=1e-3)

print("(*) Second Model (Optimizing for target entropy)")

def second_model(weights, target):
    probs = weights_to_probs(weights)
    return np.power(entropy(probs) - target, 2)

key, subkey = random.split(key)
weights = uniform(key, (n,))
target = np.log2(n) / 2
g = jit(grad(second_model))
for epoch in range(epochs):
    weights += -g(weights, target)

print(weights_to_probs(weights), first_model(weights), "(target: " + str(target) + ")")
assert np.isclose(second_model(weights, target), 0, atol=1e-3)

def kldiv(p, q):
    ratios = np.divide(p, q)
    logs = np.log2(ratios)
    logs = np.nan_to_num(logs)
    infos = np.multiply(probs, logs)
    return np.sum(infos)

assert kldiv(probs, probs) == 0
print("(*) Third Model (Minimizing KL-divergence)")

def third_model(p_weights, q_weights):
    p = weights_to_probs(p_weights)
    q = weights_to_probs(q_weights)
    return kldiv(p, q)

key, subkey = random.split(key)
p_weights = uniform(key, (n,))
key, subkey = random.split(key)
q_weights = uniform(key, (n,))

g = jit(grad(third_model))
for epoch in range(epochs):
    p_weights += g(p_weights, q_weights)

print(weights_to_probs(p_weights))
print(weights_to_probs(q_weights))
print(third_model(p_weights, q_weights))

print("(*) Fourth Model (Optimizing for target KL-divergence)")

def fourth_model(p_weights, q_weights, target):
    p = weights_to_probs(p_weights)
    q = weights_to_probs(q_weights)
    return np.power(kldiv(p, q) - target, 2)

key, subkey = random.split(key)
p_weights = uniform(key, (n,))
key, subkey = random.split(key)
q_weights = uniform(key, (n,))
target = 0.05

g = jit(grad(fourth_model))
for epoch in range(epochs):
    p_weights += -g(p_weights, q_weights, target)

print(weights_to_probs(p_weights))
print(weights_to_probs(q_weights))
print(third_model(p_weights, q_weights))
