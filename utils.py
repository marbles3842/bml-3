import jax.numpy as jnp
import numpy as np
import os 
from jax import random, lax, vmap


def metropolis(log_target,
               num_params: int,
               tau: float,
               num_iter: int,
               theta_init: jnp.ndarray = None,
               seed: int = 0):
    """
    Metropolis–Hastings sampler in JAX returning (chain, accepts).

    Args:
      log_target : fn(theta) → log unnormalized density
      num_params : dimensionality of theta
      tau        : proposal stddev
      num_iter   : how many MH steps to run
      theta_init : initial theta (shape (num_params,)), defaults to zeros
      seed       : RNG seed

    Returns:
      thetas     : array, shape (num_iter+1, num_params)
      accepts    : array of 0/1 ints, shape (num_iter,)
    """
    key = random.PRNGKey(seed)
    if theta_init is None:
        theta_init = jnp.zeros((num_params,))

    def step(carry, _):        
        theta_current, logp_current, key = carry
        key, key_proposal, key_accept = random.split(key, num=3)

        theta_star = theta_current + tau * random.normal(key_proposal, (num_params,))
        logp_star = log_target(theta_star)
        
        log_r = logp_star - logp_current
        A = jnp.exp(jnp.minimum(log_r, 0.0))

        accept_condition = random.uniform(key_accept) < A

        theta_next = jnp.where(accept_condition, theta_star, theta_current)
        logp_next  = jnp.where(accept_condition, logp_star, logp_current)

        return (theta_next, logp_next, key), (theta_next, accept_condition.astype(jnp.int32))

    _, (thetas_tail, accepts) = lax.scan(
        step,
        (theta_init, log_target(theta_init), key),
        None,
        length=num_iter
    )
    
    thetas = jnp.vstack([theta_init[None, :], thetas_tail])
    assert thetas.shape == (num_iter+1, num_params), f'The shape of thetas was expected to be ({num_iter+1}, {num_params}), but the actual shape was {thetas.shape}. Please check your code.'

    return thetas, accepts


def metropolis_multiple_chains(log_target, num_params, num_chains, tau, num_iter, theta_init, seeds, warm_up=0):
    """ Runs multiple Metropolis-Hastings chains. The i'th chain should be initialized using the i'th vector in theta_init, i.e. theta_init[i, :]

    Arguments:
        log_target:         function for evaluating the log joint distribution
        num_params:         number of parameters of the joint distribution (integer)
        num_chains:         number of MCMC chains
        tau:                variance of Gaussian proposal distribution (positive real)
        num_iter:           number of iterations for each chain (integer)
        theta_init:         array of initial values (jnp.array with shape (num_chains, num_params))        
        seeds:              seed for each chain (jnp.array with shape (num_chains))
        warm_up:            number of warm up samples to be discarded
    
    returns:
        thetas              jnp.array of samples from each chain after warmup (shape: num_chains x (num_iter + 1 - warm_up))
        accept_rates        jnp.array of acceptances rate for each chain (shapes: num_chains)
    
     """
    
    assert theta_init.shape == (num_chains, num_params), "theta_init seems to have the wrong dimensions. Plaese check your code."
    assert seeds.shape == (num_chains,), "seeds seem to have the wrong dimensions. Plaese check your code."
    
    metropolis_single_chain = lambda t, s: metropolis(log_target, num_params, tau, num_iter, t, s)
    thetas, accept_rates = vmap(metropolis_single_chain)(theta_init, seeds)
    
    thetas = thetas[:, warm_up:, :]
    
    assert thetas.shape == (num_chains, num_iter+1-warm_up, num_params), f"The expected shape of chains is ({num_chains}, {num_iter+1-warm_up}, {num_params}) corresponding to (num_chains, num_iter+1-warm_up), but the actual shape is {thetas.shape}. Check your implementation."
    assert len(accept_rates) == num_chains
    return thetas, accept_rates


def save_array(arr, filename):
    """ Saves a JAX array to the filename in the data package """
    
    path = os.path.join('assignment3', 'data', f'{filename}')
    np.save(path, np.array(arr))
    
def load_array(filename):
    """ Loads a JAX array from the filename in the data package """

    path = os.path.join('assignment3', 'data', f'{filename}')
    loaded_arr = np.load(path)
    return jnp.array(loaded_arr)