import tensorflow as tf
import numpy as np
import core as cr
from buffer import replay_buffer
from ou_noise import OU_noise
from tensorboardX import SummaryWriter

class DDPG:
    def __init__(self):
        self.sess = tf.Session()
        self.memory = replay_buffer(max_length=1e5)
        self.tau = 0.995
        self.gamma = 0.99
        self.state_size = 33
        self.output_size = 4
        self.action_limit = 1.0
        self.hidden = [400, 300]
        self.batch_size = 100
        self.pi_lr = 1e-4
        self.q_lr = 1e-4
        self.noise = OU_noise(self.output_size, 1)

        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
            cr.placeholders(self.state_size, self.output_size, self.state_size, None, None)

        with tf.variable_scope('main'):
            self.pi, self.q, self.q_pi = cr.mlp_actor_critic(self.x_ph,
                self.a_ph, self.hidden, activation=tf.nn.relu, output_activation=tf.tanh,
                output_size=self.output_size, action_limit=self.action_limit)

        with tf.variable_scope('target'):
            self.pi_targ, _, self.q_pi_targ = cr.mlp_actor_critic(self.x2_ph,\
                self.a_ph, self.hidden, activation=tf.nn.relu, output_activation=tf.tanh,
                output_size=self.output_size, action_limit=self.action_limit)

        self.target = tf.stop_gradient(self.r_ph + self.gamma * (1 - self.d_ph) * self.q_pi_targ)
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.v_loss = tf.reduce_mean((self.q - self.target) ** 2) * 0.5
        self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr)
        self.v_optimizer = tf.train.AdamOptimizer(self.q_lr)
        self.pi_train = self.pi_optimizer.minimize(self.pi_loss, var_list=cr.get_vars('main/pi'))
        self.v_train = self.v_optimizer.minimize(self.v_loss, var_list=cr.get_vars('main/q'))

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

        writer = SummaryWriter('runs/ddpg')
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
    agent = DDPG()
    agent.run()