import tensorflow as tf
import numpy as np
import core as cr
from buffer import replay_buffer
from tensorboardX import SummaryWriter
from ou_noise import OU_noise
import gym

class SAC:
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
        self.noise_clip = 0.1
        self.alpha = 1e-5
        self.num_worker = 20
        self.noise = OU_noise(self.output_size, self.num_worker)
        
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
            cr.placeholders(self.state_size, self.output_size, self.state_size, None, None)

        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = \
                cr.sac_mlp_actor_critic(
                    x=self.x_ph,
                    a=self.a_ph,
                    hidden=self.hidden,
                    activation=tf.nn.relu,
                    output_activation=tf.tanh,
                    output_size=self.output_size,
                    action_limit=self.action_limit
                )
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, self.v_targ = \
                cr.sac_mlp_actor_critic(
                    x=self.x2_ph,
                    a=self.a_ph,
                    hidden=self.hidden,
                    activation=tf.nn.relu,
                    output_activation=tf.tanh,
                    output_size=self.output_size,
                    action_limit=self.action_limit
                )

        self.pi_params = cr.get_vars('main/pi')
        self.value_params = cr.get_vars('main/q') + cr.get_vars('main/v')

        self.min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)
        self.q_backup = tf.stop_gradient(self.r_ph + self.gamma * (1 - self.d_ph) * self.v_targ)
        self.v_backup = tf.stop_gradient(self.min_q_pi - self.alpha * self.logp_pi)

        self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.q1_pi)
        self.q1_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q1) ** 2)
        self.q2_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q2) ** 2)
        self.v_loss  = 0.5 * tf.reduce_mean((self.v_backup - self.v) ** 2)
        self.value_loss = self.q1_loss + self.q2_loss + self.v_loss

        self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr)
        self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=self.pi_params)

        self.value_optimizer = tf.train.AdamOptimizer(self.q_lr)
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = self.value_optimizer.minimize(self.value_loss, var_list=self.value_params)

        with tf.control_dependencies([self.train_value_op]):
            self.target_update = tf.group([tf.assign(v_targ, self.tau*v_targ + (1-self.tau)*v_main)
                                  for v_main, v_targ in zip(cr.get_vars('main'), cr.get_vars('target'))])

        self.step_ops = [self.pi_loss, self.q1_loss, self.q2_loss, self.v_loss, self.q1, self.q2,
                         self.v,       self.logp_pi, self.train_pi_op, self.train_value_op, self.target_update]
    
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

        self.sess.run(self.step_ops, feed_dict=feed_dict)

    def get_action(self, state, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: [state]})[0]

    def test(self):
        env = gym.make('Pendulum-v0')
        while True:
            state = env.reset()
            done = False
            while not done:
                env.render()
                action = self.get_action(state, 0)
                state, _, done,_ = env.step(action)

    def run(self):
        from mlagents.envs import UnityEnvironment

        writer = SummaryWriter()
        num_worker = self.num_worker
        state_size = self.state_size
        output_size = self.output_size
        ep = 0
        train_size = 5

        env = UnityEnvironment(file_name='env/training', worker_id=1)
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        initial_observation = env.reset()

        step = 0
        start_steps = 100000

        states = np.zeros([num_worker, state_size])
        for i in range(start_steps):
            actions = np.clip(np.random.randn(num_worker, output_size), -self.action_limit, self.action_limit)
            actions += self.noise.sample()
            env_info = env.step(actions)[default_brain]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for s, ns, r, d, a in zip(states, next_states, rewards, dones, actions):
                self.memory.append(s, ns, r, d, a)
            states = next_states
            if dones[0]:
                self.noise.reset()
            if i % train_size == 0:
                if len(self.memory.memory) > self.batch_size:
                    self.update()
            print('data storing :', float(i / start_steps))

        while True:
            ep += 1
            states = np.zeros([num_worker, state_size])
            terminal = False
            score = 0
            while not terminal:
                step += 1
                '''
                if step > start_steps:
                    actions = [self.get_action(s) for s in states]
                    action_random = 'False'
                else:
                    actions = np.clip(np.random.randn(num_worker, output_size), -self.action_limit, self.action_limit)
                    action_random = 'True'
                '''
                actions = [self.get_action(s) for s in states]
                action_random = 'False'
            
                env_info = env.step(actions)[default_brain]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                terminal = dones[0]

                for s, ns, r, d, a in zip(states, next_states, rewards, dones, actions):
                    self.memory.append(s, ns, r, d, a)

                score += sum(rewards)

                states = next_states

                if len(self.memory.memory) > self.batch_size:
                    if step % train_size == 0:
                        self.update()

            if ep < 1000:
                print('step : ', step, '| start steps : ', start_steps, '| episode :', ep, '| score : ', score, '| memory size', len(self.memory.memory), '| action random : ', action_random)
                writer.add_scalar('data/reward', score, ep)
                writer.add_scalar('data/memory_size', len(self.memory.memory), ep)

    
if __name__ == '__main__':
    agent = SAC()
    agent.run()