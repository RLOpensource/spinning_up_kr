import tensorflow as tf
import numpy as np
import core as cr
import gym
from tensorboardX import SummaryWriter

class TRPO:
    def __init__(self):
        self.sess = tf.Session()
        self.gamma = 0.99
        self.lamda = 0.95
        self.state_size = 33
        self.output_size = 4
        self.hidden = [400, 300]
        self.batch_size = 32
        self.pi_lr = 0.00025
        self.v_lr = 0.00025
        self.ppo_eps = 0.2
        self.epoch = 10
        self.damping_coeff = 0.1
        self.cg_iters = 10
        self.train_v_iters = 10
        self.backtrack_iters = 10
        self.EPS = 1e-8
        self.delta = 0.01
        self.algo = 'trpo'
        self.backtrack_coeff = 0.8

        self.x_ph, self.a_ph, self.adv_ph, self.target_ph, self.logp_old_ph = \
            cr.placeholders(self.state_size, self.output_size, None, None, None)

        self.pi, self.logp, self.logp_pi, self.info, self.info_phs, self.d_kl, self.v = \
            cr.trpo_mlp_actor_critic(
                x=self.x_ph,
                a=self.a_ph,
                hidden=self.hidden,
                activation=tf.nn.relu,
                output_activation=None,
                output_size=self.output_size
            )
        
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.target_ph, self.logp_old_ph] + cr.values_as_sorted_list(self.info_phs)
        self.get_action_ops = [self.pi, self.v, self.logp_pi] + cr.values_as_sorted_list(self.info)

        self.ratio = tf.exp(self.logp - self.logp_old_ph)
        self.pi_loss = -tf.reduce_mean(self.ratio * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.v - self.target_ph) ** 2)

        self.train_vf = tf.train.AdamOptimizer(learning_rate=self.v_lr).minimize(self.v_loss)

        self.pi_params = cr.get_vars('pi')
        self.gradient = cr.flat_grad(self.pi_loss, self.pi_params)
        self.v_ph, self.hvp = cr.hessian_vector_product(self.d_kl, self.pi_params)
        if self.damping_coeff > 0:
            self.hvp += self.damping_coeff * self.v_ph
        self.get_pi_params = cr.flat_concat(self.pi_params)
        self.set_pi_params = cr.assign_params_from_flat(self.v_ph, self.pi_params)

        self.sess.run(tf.global_variables_initializer())

    def cg(self, Ax, b):
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + self.EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update(self, state, action, target, adv, logp_old, old_info):
        zip_ph = [state, action, target, adv, logp_old, old_info[:, 0], old_info[:, 1]]
        inputs = {k:v for k,v in zip(self.all_phs, zip_ph)}
        Hx = lambda x : self.sess.run(self.hvp, feed_dict={**inputs, self.v_ph: x})
        g, pi_l_old, v_l_old = self.sess.run([self.gradient, self.pi_loss, self.v_loss], feed_dict=inputs)

        x = self.cg(Hx, g)
        alpha = np.sqrt(2*self.delta/(np.dot(x, Hx(x)) + self.EPS))
        old_params = self.sess.run(self.get_pi_params)

        if self.algo == 'npg':
            kl, pi_l_new = self.set_and_eval(1., inputs, old_params, alpha, x)

        elif self.algo == 'trpo':
            for j in range(self.backtrack_iters):
                kl, pi_l_new = self.set_and_eval(self.backtrack_coeff ** j, inputs, old_params, alpha, x)
                if kl <= self.delta and pi_l_new <= pi_l_old:
                    break
                if j == (self.backtrack_iters - 1):
                    kl, pi_l_new = self.set_and_eval(0., inputs, old_params, alpha, x)
        
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_vf, feed_dict=inputs)
        v_l_new = self.sess.run(self.v_loss, feed_dict=inputs)

        return pi_l_old, v_l_old, kl, pi_l_new - pi_l_old, v_l_new, v_l_old


    def set_and_eval(self, step, inputs, old_params, alpha, x):
        self.sess.run(self.set_pi_params, feed_dict={self.v_ph: old_params - alpha * x * step})
        return self.sess.run([self.d_kl, self.pi_loss], feed_dict=inputs)

    def get_action(self, state):
        agent_outs = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: [state]})
        a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1][0], agent_outs[2][0], agent_outs[3:]
        return a, v_t, logp_t, info_t

    def run(self):
        from mlagents.envs import UnityEnvironment

        writer = SummaryWriter('runs/trpo')
        num_worker = 20
        state_size = self.state_size
        output_size = self.output_size
        n_step = 128
        ep = 0
        score = 0

        env = UnityEnvironment(file_name='env/training', worker_id=0)
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        initial_observation = env.reset()

        states= np.zeros([num_worker, state_size])
        
        while True:
            values_list, states_list, actions_list, dones_list, logp_ts_list, rewards_list, infos_list = \
                [], [], [], [], [], [], []
            for _ in range(n_step):
                inference = [self.get_action(s) for s in states]
                actions = [inf[0] for inf in inference]
                values = [inf[1] for inf in inference]
                logp_ts = [inf[2] for inf in inference]
                info_ts = [inf[3] for inf in inference]

                env_info = env.step(actions)[default_brain]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                score += sum(rewards)

                states_list.append(states)
                values_list.append(values)
                actions_list.append(actions)
                dones_list.append(dones)
                logp_ts_list.append(logp_ts)
                rewards_list.append(rewards)
                infos_list.append(info_ts)

                states = next_states

                if dones[0]:
                    ep += 1
                    if ep < 1000:
                        writer.add_scalar('data/reward', score, ep)
                        print(ep, score)
                    score = 0

            inference = [self.get_action(s) for s in states]
            values = [inf[1] for inf in inference]
            values_list.append(values)
            values_list = np.stack(values_list).transpose([1, 0])

            current_value_list = values_list[:, :-1]
            next_value_list = values_list[:, 1:]

            states_list = np.stack(states_list).transpose([1, 0, 2]).reshape([-1, state_size])
            actions_list = np.stack(actions_list).transpose([1, 0, 2]).reshape([-1, output_size])
            dones_list = np.stack(dones_list).transpose([1, 0]).reshape([-1])
            logp_ts_list = np.stack(logp_ts_list).transpose([1, 0]).reshape([-1])
            rewards_list = np.stack(rewards_list).transpose([1, 0]).reshape([-1])
            current_value_list = np.stack(current_value_list).reshape([-1])
            next_value_list = np.stack(next_value_list).reshape([-1])
            infos_list = np.stack(infos_list).transpose([1, 0, 2, 3, 4]).reshape([-1, 2, output_size])
            
            adv_list, target_list = [], []
            for idx in range(num_worker):
                start_idx = idx * n_step
                end_idx = (idx + 1) * n_step
                adv, target = cr.get_gaes(
                    rewards_list[start_idx : end_idx],
                    dones_list[start_idx : end_idx],
                    current_value_list[start_idx : end_idx],
                    next_value_list[start_idx : end_idx],
                    self.gamma,
                    self.lamda,
                    True
                )
                adv_list.append(adv)
                target_list.append(target)
            adv_list = np.stack(adv_list).reshape([-1])
            target_list = np.stack(target_list).reshape([-1])
            self.update(states_list, actions_list, target_list, adv_list, logp_ts_list, infos_list)


if __name__ == '__main__':
    agent = TRPO()
    agent.run()