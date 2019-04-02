import tensorflow as tf
import numpy as np
import copy

EPS = 1e-8

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def actor_mlp_without_action(x, hidden, output_size, activation, output_activation):
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    return tf.layers.dense(inputs=x, units=output_size, activation=output_activation)

def critic_mlp_with_action(x, a, hidden, activation, output_activation):
    x = tf.concat([x, a], axis=-1)
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    return tf.layers.dense(inputs=x, units=1, activation=output_activation)

def critic_mlp_without_action(x, hidden, activation, output_activation):
    for h in hidden:
        x = tf.layers.dense(inputs=x, units=h, activation=activation)
    return tf.layers.dense(inputs=x, units=1, activation=output_activation)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

## for trpo
def trpo_mlp_actor_critic(x, a, hidden, activation, output_activation, output_size):
    with tf.variable_scope('pi'):
        mu = actor_mlp_without_action(x, hidden, output_size, activation, output_activation)
        log_std = tf.ones(tf.shape(mu)) * -1.0
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        old_mu_ph, old_log_std_ph = placeholders(output_size, output_size)
        d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)
        info = {'mu': mu, 'log_std': log_std}
        info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    with tf.variable_scope('v'):
        v = tf.squeeze(critic_mlp_without_action(x, hidden, activation, None), axis=1)

    return pi, logp, logp_pi, info, info_phs, d_kl, v

## for sac
def sac_mlp_actor_critic(x, a, hidden, activation, output_activation,
                            output_size, action_limit):
    log_std_min = -20.0
    log_std_max = 2.0
    with tf.variable_scope('pi'):
        net = actor_mlp_without_action(x, hidden[:-1], hidden[-1], activation, activation)
        mu = tf.layers.dense(inputs=net, units=output_size, activation=output_activation)
        log_std = tf.layers.dense(inputs=net, units=output_size, activation=tf.tanh)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    with tf.variable_scope('q1'):
        q1 = tf.squeeze(critic_mlp_with_action(x, a, hidden, activation, None), axis=1)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = tf.squeeze(critic_mlp_with_action(x, pi, hidden, activation, None), axis=1)
    with tf.variable_scope('q2'):
        q2 = tf.squeeze(critic_mlp_with_action(x, a, hidden, activation, None), axis=1)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = tf.squeeze(critic_mlp_with_action(x, pi, hidden, activation, None), axis=1)
    with tf.variable_scope('v'):
        v = tf.squeeze(critic_mlp_without_action(x, hidden, activation, None), axis=1)
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

## for ppo
def ppo_mlp_actor_critic(x, a, hidden, activation, output_activation,
                            output_size):
    with tf.variable_scope('pi'):
        mu = actor_mlp_without_action(x, hidden, output_size, activation, output_activation)
        log_std = tf.ones(tf.shape(mu)) * -1.0
        #log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(output_size, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)

    with tf.variable_scope('v'):
        v = tf.squeeze(critic_mlp_without_action(x, hidden, activation, None), axis=1)
    
    return pi, logp, logp_pi, v

## for ddpg
def mlp_actor_critic(x, a, hidden, activation, output_activation,
                        output_size, action_limit):
    with tf.variable_scope('pi'):
        pi = action_limit * actor_mlp_without_action(x, hidden, output_size, activation,
                                               output_activation)

    with tf.variable_scope('q'):
        q = tf.squeeze(critic_mlp_with_action(x, a, hidden, activation, None), axis=1)
    
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(critic_mlp_with_action(x, pi, hidden, activation, None), axis=1)

    return pi, q, q_pi

## for td3
def td3_mlp_actor_critic(x, a, hidden, activation, output_activation,
                        output_size, action_limit):
    with tf.variable_scope('pi'):
        pi = action_limit * actor_mlp_without_action(x, hidden, output_size, activation,
                                               output_activation)

    with tf.variable_scope('q2'):
        q2 = tf.squeeze(critic_mlp_with_action(x, a, hidden, activation, None), axis=1)

    with tf.variable_scope('q1'):
        q1 = tf.squeeze(critic_mlp_with_action(x, a, hidden, activation, None), axis=1)
    
    with tf.variable_scope('q1', reuse=True):
        q1_pi = tf.squeeze(critic_mlp_with_action(x, pi, hidden, activation, None), axis=1)

    return pi, q1, q2, q1_pi