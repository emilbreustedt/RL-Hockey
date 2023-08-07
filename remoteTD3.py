import numpy as np

from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
import laserhockey.hockey_env as h_env
import torch

from TD3 import TD3Agent
        
class RemoteTD3(TD3Agent, RemoteControllerInterface):
    def __init__(self):
        config = {
            "name" : "name",
            "agent_type" : "TD3",
            "env_type" : "hockey",
            "test" : True,
            "render" : False,
            "episodes" : 400,
            "mode" : "normal",
            "eps" : 0.0,
            "discount":0.99,
            "update_target_every":100,
            "update_policy_every":2,
            "hidden_sizes_actor" : [256,256],
            "hidden_sizes_critic" : [256,256],
            "iter_fit" : 1,
            "batch_size" : 256,
            "smoothing_std"  : 0.0001,
            "smoothing_clip" : 0.0002,
            "checkpoint1" : None,
            "checkpoint2" : None,
            "learning_rate_critic": 0.001,
            "learning_rate_actor": 0.001,
            "buffer_size" : int(1e6),
            "theta" : 0.005,
            "prio_replay" : False,
            "exp_decay" : 1,
            "cdq" : True
        }
        name = "weak 4th"
        mode = "weak"
        config["checkpoint1"] = f'./results/{config["agent_type"]}_hockey_{name}_{mode}_agent.pth'
        env = h_env.HockeyEnv()
        TD3Agent.__init__(self,env.observation_space, env.action_space, discount=config["discount"], buffer_size=config["buffer_size"], eps=config["eps"],
                                  update_target_every=config["update_target_every"], update_policy_every=config["update_policy_every"], 
                                  hidden_sizes_actor=config["hidden_sizes_actor"],hidden_sizes_critic=config["hidden_sizes_critic"],
                                  smoothing_std=config["smoothing_std"], smoothing_clip=config["smoothing_clip"], batch_size=config["batch_size"],
                                  learning_rate_actor=config["learning_rate_actor"], learning_rate_critic=config["learning_rate_critic"], 
                                  theta=config["theta"], cdq=config["cdq"])
        self.restore_state(torch.load(config["checkpoint1"]))
        env.close()
        RemoteControllerInterface.__init__(self, identifier='TD3')
        
        
        
    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:
        return self.act(obs)
               
if __name__ == '__main__':
    controller = RemoteTD3()
    # Play n (None for an infinite amount) games and quit
    client = Client(username='Q-Tips', # Testuser
                    password='AiShaiL9ch',
                    controller=controller, 
                    output_path='/remote_games', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)