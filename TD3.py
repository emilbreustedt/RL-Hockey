import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

import memory as mem
from feedforward import Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))

class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)

class TD3Agent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate_actor": 0.0001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 100,
            "update_policy_every": 100,
            "use_target_net" : True,
            "cdq": True,
            "smoothing_std": 0.0005,
            "smoothing_clip": 0.00025,
            "theta" : 0.005
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']

        self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Networks
        self.Q1 = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"])
        self.Q2 = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"])
        # target Q Networks
        self.Q_target1 = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0)
        self.Q_target2 = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0)
        # policy Networks
        self.policy1 = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh())        
        self.policy2 = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh())
        # target policy Networks
        self.policy_target1 = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh())
        self.policy_target2 = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh())

        self._copy_nets()

        self.optimizer1=torch.optim.Adam(self.policy1.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.optimizer2=torch.optim.Adam(self.policy2.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target1.load_state_dict(self.Q1.state_dict())
        self.Q_target2.load_state_dict(self.Q2.state_dict())
        self.policy_target1.load_state_dict(self.policy1.state_dict())
        self.policy_target2.load_state_dict(self.policy2.state_dict())

    def sliding_update(self):
        theta = self._config["theta"]
        #print("q1 ",next(self.Q1.parameters())[0,0])
        #print("qt1 before ", next(self.Q_target1.parameters())[0,0])
        # sliding critic target update
        Q2_net = self.Q2.parameters()
        Q_target1 = self.Q_target1.parameters()
        Q_target2 = self.Q_target2.parameters()
        for q1 in self.Q1.parameters():
            q2 = next(Q2_net)
            qt1 = next(Q_target1)
            qt2 = next(Q_target2)
            with torch.no_grad():
                qt1.copy_((1-theta)*qt1 + theta*q1)
                qt2.copy_((1-theta)*qt2 + theta*q2)
        
        #print("qt1 after ", next(self.Q_target1.parameters())[0,0])
        # sliding actor target update
        P2_net = self.policy2.parameters()
        P_target1 = self.policy_target1.parameters()
        P_target2 = self.policy_target2.parameters()
        for p1 in self.policy1.parameters():
            p2 = next(P2_net)
            pt1 = next(P_target1)
            pt2 = next(P_target2)
            with torch.no_grad():
                pt1.copy_((1-theta)*pt1 + theta*p1)
                pt2.copy_((1-theta)*pt2 + theta*p2)
        
    def remote_act(self, obs : np.ndarray,) -> np.ndarray:
        return self.act(obs)
    
    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        #action = self.policy1.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        #print(self.policy1.predict(observation).shape, np.random.normal(0.0,eps,self._action_n,1).shape)
        action = self.policy1.predict(observation) + np.random.normal(0.0,eps,self._action_n)
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def policy_smoothing(self, actions):
        actions_smoothed = actions + torch.clamp(torch.normal(0,self._config["smoothing_std"], actions.size()), min=-self._config["smoothing_clip"], max=self._config["smoothing_clip"])
        return actions_smoothed
        

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q1.state_dict(), self.Q2.state_dict(), self.policy1.state_dict(), self.policy2.state_dict())

    def restore_state(self, state):
        self.Q1.load_state_dict(state[0])
        self.Q2.load_state_dict(state[1])
        self.policy1.load_state_dict(state[2])
        self.policy2.load_state_dict(state[3])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        cdq = self._config["cdq"]
        self.train_iter+=1
        grads = 0
        #if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
        #    self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
            

            # 
            action1 = self.policy_smoothing(self.policy_target1.forward(s_prime))
            if cdq:
                action2 = self.policy_smoothing(self.policy_target2.forward(s_prime))
                q_prime1 = torch.min(torch.cat((
                    self.Q_target1.Q_value(s_prime, action1),
                    self.Q_target2.Q_value(s_prime, action1)),dim=1),dim=1, keepdim=True)[0]
                q_prime2 = torch.min(torch.cat((
                    self.Q_target1.Q_value(s_prime, action2),
                    self.Q_target2.Q_value(s_prime, action2)),dim=1),dim=1, keepdim=True)[0]
                
            else:
                q_prime1 = self.Q_target1.Q_value(s_prime, action1)
                
            gamma=self._config['discount']
            
            td_target1 = rew + gamma * (1.0-done) * q_prime1
            fit_loss1 = self.Q1.fit(s, a, td_target1)
            fit_loss2 = 0
            if cdq:
                td_target2 = rew + gamma * (1.0-done) * q_prime2
                fit_loss2 = self.Q2.fit(s, a, td_target2)

            # optimize actor objective delayed
            if self.train_iter % self._config["update_policy_every"] == 0:
                self.sliding_update()
                
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                q1 = self.Q1.Q_value(s, self.policy1.forward(s))
                actor_loss1 = -torch.mean(q1)
                actor_loss1.backward()
                self.optimizer1.step()
                actor_loss2 = 0
                if cdq:
                    q2 = self.Q2.Q_value(s, self.policy2.forward(s))
                    actor_loss2 = -torch.mean(q2)
                    actor_loss2.backward()
                    self.optimizer2.step()
                    actor_loss2 = actor_loss2.item()
                #for k in self.policy1.parameters():
                    #print('===========\ngradient:\n----------\nmin:{}  max{}'.format(torch.min(k.grad),torch.max(k.grad)))
                    #grads += torch.sum(torch.abs(k.grad))
                #print(actor_loss1)
                losses.append((fit_loss1, actor_loss1.item(), fit_loss2, actor_loss2))
            else:
                losses.append((fit_loss1, None, fit_loss2, None))
                
            
        #if self.train_iter % self._config["update_policy_every"] == 0:
            #print(grads)
        
        return losses


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    random_seed = opts.seed
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ddpg = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        ddpg.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = ddpg.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            ddpg.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            if done or trunc: break

        losses.extend(ddpg.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(ddpg.state(), f'./results/DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()

if __name__ == '__main__':
    main()
