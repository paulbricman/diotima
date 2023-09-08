import jax

jax.distributed.initialize()
print(jax.devices())
print(jax.local_devices())
print(jax.device_count())
print(jax.local_device_count())
