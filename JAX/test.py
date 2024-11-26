import jax.numpy as jnp
from jax.lib import xla_bridge
import jax

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
devices = jax.local_devices()
print("ADRIAN",devices) 


print(jax.devices())

def selu(x, alpha=1.67, lmbda=1.05):
      return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))
