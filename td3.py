import tensorflow as tf
import numpy as np
import core as cr
from buffer import replay_buffer
from ou_noise import OU_noise
from tensorboardX import SummaryWriter
import gym

class TD3:
    def __init__(self):
        self.sess = tf.Session()
        self.state_size = 33
        self.output_size = 4
        self.tau = 0.995
        self.gamma = 0.99
        self.hidden = [400, 300]
        self.batch_size = 64
        self.pi_lr = 1e-3
        self.q_lr = 1e-3
        self.action_limit = 1.0
        self.memory = replay_buffer(1e5)
        self.target_noise = 0.2
        self.noise = OU_noise(self.output_size, 1)
        self.noise_clip = 0.1
        
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
            cr.placeholders(self.state_size, self.output_size, self.state_size, None, None)

        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = cr.td3_mlp_actor_critic(
                x=self.x_ph,
                a=self.a_ph,
                hidden=self.hidden,
                activation=tf.nn.relu,
                output_activation=tf.tanh,
                output_size=self.output_size,
                action_limit=self.action_limit
            )

        with tf.variable_scope('target'):
            self.pi_targ, _, _, _ = cr.td3_mlp_actor_critic(
                x=self.x2_ph,
                a=self.a_ph,
                hidden=self.hidden,
                activation=tf.nn.relu,
                output_activation=tf.tanh,
                output_size=self.output_size,
                action_limit=self.action_limit
            )

        with tf.variable_scope('target', reuse=True):
            self.eps = tf.random_normal(tf.shape(self.pi_targ), stddev=self.target_noise)
            self.epsilon = tf.clip_by_value(self.eps, -self.noise_clip, self.noise_clip)
            self.a_prev = self.pi_targ + self.epsilon
            self.a2 = tf.clip_by_value(self.a_prev, -self.action_limit, self.action_limit)
            _, self.q1_targ, self.q2_targ, self.q1_pi_targ = cr.td3_mlp_actor_critic(
                x=self.x2_ph,
                a=self.a2,
                hidden=self.hidden,
                activation=tf.nn.relu,
                output_activation=tf.tanh,
                output_size=self.output_size,
                action_limit=self.action_limit
            )

        self.pi_params = cr.get_vars('main/pi')
        self.q_params = cr.get_vars('main/q')

        self.min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)
        self.backup = tf.stop_gradient(self.r_ph + self.gamma*(1-self.d_ph)*self.min_q_targ)
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        self.q1_loss = tf.reduce_mean((self.q1-self.backup)**2)
        self.q2_loss = tf.reduce_mean((self.q2-self.backup)**2)
        self.v_loss = self.q1_loss + self.q2_loss
        
        self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr)
        self.q_optimizer = tf.train.AdamOptimizer(self.q_lr)
        self.pi_train = self.pi_optimizer.minimize(self.pi_loss, var_list=self.pi_params)
        self.v_train = self.pi_optimizer.minimize(self.v_loss, var_list=self.q_params)

        self.target_update = tf.group([tf.assign(v_targ, self.tau * v_targ + (1 - self.tau) * v_main)
                                    for v_main, v_targ in zip(cr.get_vars('main'), cr.get_vars('target'))])

        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(cr.get_vars('main'), cr.get_vars('target'))])

        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.target_init)

    def update(self):
        data = self.memory.get_sample(sample_size=self.batch_size)
        feed_dict = {
            self.x_ph : data['state'],
            self.x2_ph : data['next_state'],
            self.a_ph : data['action'],
            self.r_ph : data['reward'],
            self.d_ph : data['done']
        }

        q_loss, _ = self.sess.run([self.v_loss, self.v_train], feed_dict=feed_dict)
        pi_loss, _, _ = self.sess.run([self.pi_loss, self.pi_train, self.target_update], feed_dict=feed_dict)

        return q_loss, pi_loss

    def get_action(self, state, epsilon):
        a = self.sess.run(self.pi, feed_dict={self.x_ph: [state]})
        a += epsilon * self.noise.sample()
        return np.clip(a, -self.action_limit, self.action_limit)[0]

    def test(self):
        env = gym.make('Pendulum-v0')
        while True:
            state = env.reset()
            done = False
            while not done:
                env.render()
                action = self.get_action(state, 0)
                next_state, _, done,_ = env.step(action)
                state = next_state

    def run(self):
        from mlagents.envs import UnityEnvironment

        writer = SummaryWriter('runs/td3')
        num_worker = 20
        state_size = 33
        output_size = 4
        epsilon = 1.0
        ep = 0
        train_size = 5

        env = UnityEnvironment(file_name='env/training', worker_id=0)
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        initial_observation = env.reset()

        step = 0
        score = 0

        while True:
            ep += 1
            env_info = env.reset()
            states = np.zeros([num_worker, state_size])
            terminal = False
            self.noise.reset()
            if epsilon > 0.001:
                epsilon = -ep * 0.005 + 1.0
            while not terminal:
                step += 1

                actions = [self.get_action(s, epsilon) for s in states]
                env_info = env.step(actions)[default_brain]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                terminal = dones[0]

                for s, ns, r, d, a in zip(states, next_states, rewards, dones, actions):
                    self.memory.append(s, ns, r, d, a)

                score += sum(rewards)

                states = next_states

                if step % train_size == 0:
                    self.update()

            if ep < 1000:
                print('episode :' ,ep, '| score : ', score, '| epsilon :', epsilon)
                writer.add_scalar('data/reward', score, ep)
                writer.add_scalar('data/epsilon', epsilon, ep)
                writer.add_scalar('data/memory_size', len(self.memory.memory), ep)
                score = 0


if __name__ == '__main__':
    agent = TD3()
    agent.run()
    agent.test()