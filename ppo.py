import tensorflow as tf
import numpy as np
import core as cr
import gym
from tensorboardX import SummaryWriter

class PPO:
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

        self.x_ph, self.a_ph, self.adv_ph, self.target_ph, self.logp_old_ph = \
            cr.placeholders(self.state_size, self.output_size, None, None, None)

        self.pi, self.logp, self.logp_pi, self.v = cr.ppo_mlp_actor_critic(
            x=self.x_ph,
            a=self.a_ph,
            hidden=self.hidden,
            activation=tf.nn.relu,
            output_activation=None,
            output_size=self.output_size
        )

        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.target_ph, self.logp_old_ph]
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        self.ratio = tf.exp(self.logp - self.logp_old_ph)

        self.min_adv = tf.where(self.adv_ph > 0, (1.0 + self.ppo_eps)*self.adv_ph, (1.0 - self.ppo_eps)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
        self.v_loss = tf.reduce_mean((self.target_ph - self.v) ** 2)

        self.train_pi = tf.train.AdamOptimizer(self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(self.v_lr).minimize(self.v_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)
        self.approx_ent = tf.reduce_mean(-self.logp)

        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, target, adv, logp_old):
        zip_ph = [state, action, adv, target, logp_old]
        inputs = {k:v for k,v in zip(self.all_phs, zip_ph)}
        value_loss, kl, ent = 0, 0, 0
        for i in range(self.epoch):
            _, _, v_loss, approxkl, approxent = self.sess.run([self.train_pi, self.train_v, self.v_loss, self.approx_kl, self.approx_ent], feed_dict=inputs)
            value_loss += v_loss
            kl += approxkl
            ent += approxent
        return value_loss, kl, ent

    def get_action(self, state):
        a, v, logp_t = self.sess.run(self.get_action_ops, feed_dict={
                                                            self.x_ph: [state]})
        return a[0], v[0], logp_t[0]
    
    def test(self):
        env = gym.make('BipedalWalker-v2')
        while True:
            state = env.reset()
            done = False
            while not done:
                a, _, _ = self.get_action(state)
                next_state, _, done, _ = env.step(a)
                state = next_state
    
    
    def run(self):
        from mlagents.envs import UnityEnvironment

        writer = SummaryWriter('runs/ppo')
        num_worker = 20
        state_size = 33
        output_size = 4
        n_step = 128
        ep = 0
        score = 0

        env = UnityEnvironment(file_name='env/training', worker_id=0)
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        initial_observation = env.reset()

        env_info = env.reset()
        states = np.zeros([num_worker, state_size])

        while True:
            values_list, states_list, actions_list, dones_list, logp_ts_list, rewards_list = \
                        [], [], [], [], [], []
            for _ in range(n_step):
                inference = [self.get_action(s) for s in states]
                actions = [inf[0] for inf in inference]
                values = [inf[1] for inf in inference]
                logp_ts = [inf[2] for inf in inference]

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
            
            value_loss, kl, ent = self.update(states_list, actions_list, target_list, adv_list, logp_ts_list)
        
if __name__ == '__main__':
    agent = PPO()
    agent.run()
    agent.test()