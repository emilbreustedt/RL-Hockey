import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
from TD3_helpers import *
import time
import torch
import DDPG
import TD3
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

### moving average to smooth out rewards ###
def moving_average(data, win_size):
    data = np.asarray(data)
    averages = []
    for i in range(len(data)-win_size):
        averages.append(np.sum(data[i:i+win_size])/win_size)
    return averages

    
### opponent that performs random actions
class Random_opponent():
    def __init__(self, keep_mode=True):
        self.keep_mode = keep_mode
    def act(self, obs):
        if self.keep_mode:
            return np.random.uniform(-1,1,4)
        return np.random.uniform(-1,1,3)


### function for saving train/test statistics
def save_statistics(type, config, rewards, net_losses, wins, losses, winrate):
    train_type = config["test"]*"test" + (1-config["test"])*"train"
    with open(f'./results/{type}_{config["env_type"]}_{config["name"]}_{config["mode"]}_{train_type}_stats.pkl', 'wb') as f:
        pickle.dump({"Experiment setup" : config, "Rewards": rewards, "losses": net_losses, "wins": wins, "losses":losses, "winrate": winrate}, f)

### training/testing function for gym environments ###
def train_gym(agent1, config):
    save_as1=f'./results/{config["agent_type"]}_{config["env_type"]}_{config["name"]}_{config["mode"]}_agent.pth'
    player1 = agent1
    train_losses = np.empty((0,4))
    if config["agent_type"] == "DDPG":
        train_losses = np.empty((0,2))
    if config["env_type"] == "walker":
        env = gym.make("BipedalWalker-v3", hardcore=False)
    if config["env_type"] == "pendulum":
        env = gym.make("Pendulum-v1")
    if config["env_type"] == "cheetah":
        env = gym.make("HalfCheetah-v4")
    eps = 1.0 # entirely random actions for initial 
    desc = "Training..."
    eps = config["eps"]
    if config["test"]:
        desc="Testing..."
        eps = 0.0
    obs, info = env.reset()
    d = False
    rewards = []
    ep_r = 0
    ep_steps = 0
    for i in tqdm(range(config["max_steps"]), desc=desc, unit="steps", colour="green"):
        if d or ep_steps>1000:
            rewards.append(ep_r)
            obs, info = env.reset()
            ep_r = 0
            ep_steps = 0
        if config["render"]:
            env.render()
        a1 = player1.act(obs, eps=eps)
        obsnew, r, d, _, info = env.step(a1)
        ep_r += r
        ep_steps += 1
        if not config["test"]:
            player1.store_transition((obs, a1, r, obsnew, d))
            if config["prio_replay"] and abs(r)>0:
               for k in range(5):
                   player1.store_transition((obs, a1, r, obsnew, d))
        obs=obsnew
        if not config["test"] and i>config["exp_phase"]:
            eps = config["eps"]
            loss = player1.train(config["iter_fit"])
            train_losses = np.concatenate((train_losses, np.asarray(loss)))
    env.close()
    save_statistics(config["agent_type"], config, rewards, train_losses, wins=None, losses=None, winrate=None)
    if not config["test"]:
        torch.save(player1.state(), save_as1)      
    return train_losses, rewards    

    
### training and testing for hockey environments ###
def train_hockey(agent_type, agent1, agent2, config):
    save_as1=f'./results/{agent_type}_hockey_{config["name"]}_{config["mode"]}_agent.pth'
    save_as2=f'./results/{agent_type}_hockey_{config["name"]}_{config["mode"]}_agent.pth'
    if config["mode"]=="normal" or config["mode"]=="weak" or config["mode"]=="selfplay":
        env = h_env.HockeyEnv()
    elif config["mode"]=="defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif config["mode"]=="attack":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        
    if config["mode"]=="normal":
        player2 = h_env.BasicOpponent(weak=False)
    elif config["mode"]=="weak":
        player2 = h_env.BasicOpponent()
    elif config["mode"]=="defense" or config["mode"]=="attack":
        player2 = Random_opponent()
    elif config["mode"]=="selfplay":
        player2 = agent2
        
    player1 = agent1
    obs_agent2 = env.obs_agent_two()
    if type(agent1).__name__=="TD3Agent":
        train_losses = np.empty((0,4))
    else:
        train_losses = np.empty((0,2))
    rewards = []
    wins, losses, draws, rewards = 0, 0, 0, []
    eps = config["eps"]
    eps = 1.0
    if config["retrain"]:
        eps = 0.0 # entirely random actions for initial 
    desc = "Training..."
    if config["test"]:
        desc="Testing..."
        eps = 0.0
        config["max_steps"] = int(1e6)
    obs, info = env.reset()
    d = False
    rewards = []
    ep_r = 0
    ep_num = 0
    for i in tqdm(range(config["max_steps"]), desc=desc, unit="steps", colour="green"):
        if d:
            ep_num += 1
            rewards.append(ep_r)
            obs, info = env.reset()
        if config["render"]:
            env.render()
        a1 = player1.act(obs, eps=eps)
        a2 = player2.act(obs_agent2)
        obsnew, r, d, _, info = env.step(np.hstack([a1,a2]))
        obs_agent2 = env.obs_agent_two()
        if info["winner"] == 1:
            wins += 1
        if info["winner"] == -1:
            losses += 1
        if d and info["winner"]==0:
            draws += 1
        if not config["test"]:
            player1.store_transition((obs, a1, r, obsnew, d))
            if config["prio_replay"] and r>0:
               for k in range(5):
                   player1.store_transition((obs, a1, r, obsnew, d))
            if config["mode"]=="selfplay":
                player2.store_transition((obs, a2, r, obsnew, d))
        obs=obsnew
        ep_r += r
        if not config["test"] and i>config["exp_phase"]:
            eps = config["eps"]
            loss = player1.train(config["iter_fit"])
            train_losses = np.concatenate((train_losses, np.asarray(loss)))
        if config["test"] and ep_num == config["episodes"]:
            break
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print(f'Draws: {draws}')
    winrate = wins/max(1,losses)
    print(f'W/L: {winrate}')
    env.close()
    save_statistics(agent_type, config, rewards, train_losses, wins, losses, winrate)
    if not config["test"]:
        torch.save(player1.state(), save_as1)    
    return train_losses, rewards

### function to initializes training/test and plot functions ###
def init_train(config):
    agent_type = config["agent_type"]
    if config["env_type"] == "hockey":
        env = h_env.HockeyEnv()
    else:
        if config["env_type"] == "walker":
            env = gym.make("BipedalWalker-v3", hardcore=False)
        if config["env_type"] == "pendulum":
            env = gym.make("Pendulum-v1")
        if config["env_type"] == "cheetah":
            env = gym.make("HalfCheetah-v4")
    # turn on/off the respective parts of TD3 to analyze separately
    if agent_type == "CDQ":
        config["smoothing_clip"] = 0
        config["update_policy_every"] = 1
    if agent_type == "TPS":
        config["cdq"] = False
        config["update_policy_every"] = 1
    if agent_type == "DPU":
        config["cdq"] = False
        config["smoothing_clip"] = 0
    if agent_type == "TD3_PRIO":
        config["prio_replay"] = True
    
    if agent_type =="DDPG":
        agent1 = DDPG.DDPGAgent(env.observation_space, env.action_space, discount=config["discount"], buffer_size=config["buffer_size"], eps=config["eps"],
                              update_target_every=config["update_target_every"], update_policy_every=config["update_policy_every"], 
                              hidden_sizes_actor=config["hidden_sizes_actor"],hidden_sizes_critic=config["hidden_sizes_critic"],
                              smoothing_std=config["smoothing_std"], smoothing_clip=config["smoothing_clip"], batch_size=config["batch_size"],
                              learning_rate_actor=config["learning_rate_actor"], learning_rate_critic=config["learning_rate_critic"],
                                env_type=config["env_type"], ou=config["ou"])
    else:
        agent1 = TD3.TD3Agent(env.observation_space, env.action_space, discount=config["discount"], buffer_size=config["buffer_size"], eps=config["eps"],
                              update_target_every=config["update_target_every"], update_policy_every=config["update_policy_every"], 
                              hidden_sizes_actor=config["hidden_sizes_actor"],hidden_sizes_critic=config["hidden_sizes_critic"],
                              smoothing_std=config["smoothing_std"], smoothing_clip=config["smoothing_clip"], batch_size=config["batch_size"],
                              learning_rate_actor=config["learning_rate_actor"], learning_rate_critic=config["learning_rate_critic"], 
                              theta=config["theta"], cdq=config["cdq"],
                                env_type=config["env_type"], ou=config["ou"])
    agent2 = None
    if config["mode"] == "selfplay":
        agent2 = TD3.TD3Agent(env.observation_space, env.action_space, discount=config["discount"], buffer_size=config["buffer_size"], eps=config["eps"],
                              update_target_every=config["update_target_every"], update_policy_every=config["update_policy_every"], 
                              hidden_sizes_actor=config["hidden_sizes_actor"],hidden_sizes_critic=config["hidden_sizes_critic"],
                              smoothing_std=config["smoothing_std"], smoothing_clip=config["smoothing_clip"], batch_size=config["batch_size"],
                              learning_rate_actor=config["learning_rate_actor"], learning_rate_critic=config["learning_rate_critic"], 
                              theta=config["theta"], cdq=config["cdq"],
                                env_type=config["env_type"], ou=config["ou"])
    if config["checkpoint1"]:
        agent1.restore_state(torch.load(config["checkpoint1"]))
    if config["checkpoint2"]:
        agent2.restore_state(torch.load(config["checkpoint2"]))
    env.close() 
    if config["env_type"]=="hockey":
        losses_wea, rewards_wea = train_hockey(agent_type, agent1, agent2, config)
    else:
        losses_wea, rewards_wea = train_gym(agent1,config)
    rewards_wea_avg = moving_average(rewards_wea, 10)
    if not config["test"]:
        plt.figure(figsize=(3,2))
        plt.plot(rewards_wea_avg)
        plt.title(f'{type(agent1).__name__}_wea_{config["mode"]}')
        plt.show()
        plt.figure(figsize=(3,2))
        plt.plot(moving_average(losses_wea[:,0],10))
        plt.title(f'{type(agent1).__name__}_wea_{config["mode"]}')
        plt.show()
        plt.figure(figsize=(3,2))
        plt.plot(moving_average(-losses_wea[:,1][losses_wea[:,1] != np.array(None)],10))
        plt.title(f'{type(agent1).__name__}_wea_{config["mode"]}')
        plt.show()
        if agent_type =="TD3" or agent_type =="CDQ":
            plt.figure(figsize=(3,2))
            plt.plot(moving_average(losses_wea[:,2],10))
            plt.title(f'{type(agent1).__name__}_wea_{config["mode"]}')
            plt.show()
            
            plt.figure(figsize=(3,2))
            plt.plot(moving_average(-losses_wea[:,3][losses_wea[:,3] != np.array(None)],10))
            plt.title(f'{type(agent1).__name__}_wea_{config["mode"]}')
            plt.show()
