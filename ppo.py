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
        self.state_size = 24
        self.output_size = 4
        self.hidden = [128, 128, 128]
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

        env = gym.make('BipedalWalker-v2')
        n_step = 512
        rollout = 0
        ep = 0
        score = 0
        state, done = env.reset(), False
        writer = SummaryWriter()
        episode = 1000

        while True:
            rollout += 1
            values, states, actions, dones, logp_ts, rewards =\
                     [], [], [], [], [], []
        
            for _ in range(n_step):
                #env.render()
                a, v, logp_t = self.get_action(state)
                
                next_state, reward, done, _ = env.step(a)

                score += reward

                states.append(state)
                values.append(v)
                actions.append(a)
                dones.append(done)
                logp_ts.append(logp_t)
                rewards.append(reward)

                state = next_state

                if done:
                    ep += 1
                    writer.add_scalar('data/reward', score, ep)
                    print(ep, score)
                    state, done = env.reset(), False
                    score = 0
                    if ep == episode:
                        while True:
                            state = env.reset()
                            done = False
                            while not done:
                                env.render()
                                a, _, _ = self.get_action(state)
                                next_state, _, done, _ = env.step(a)
                                state = next_state
            
            _, v, _ = self.get_action(state)
            values.append(v)
            states = np.stack(states)
            values = np.stack(values)
            actions = np.stack(actions)
            dones = np.stack(dones)
            logp_ts = np.stack(logp_ts)
            rewards = np.stack(rewards)
    
            current_value = values[:-1]
            next_value = values[1:]
            
            adv, target = cr.get_gaes(rewards, dones, current_value, next_value,
                                        self.gamma, self.lamda, True)
            value_loss, kl, ent = self.update(states, actions, target, adv, logp_ts)

            writer.add_scalar('data/value_loss', value_loss, rollout)
            writer.add_scalar('data/kl', kl, rollout)
            writer.add_scalar('data/ent', ent, rollout)

            values, states, actions, dones, logp_ts, rewards =\
                     [], [], [], [], [], []

if __name__ == '__main__':
    agent = PPO()
    agent.run()
    agent.test()