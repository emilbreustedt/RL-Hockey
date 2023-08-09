import numpy as np

from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
import laserhockey.hockey_env as h_env

from SAC import Agent
        
class RemoteSAC(Agent, RemoteControllerInterface):
    def __init__(self):
        env = h_env.HockeyEnv()
        self.agent = Agent(
            input_dims=env.observation_space.shape, 
            env=env)
        self.agent.load_models()
        env.close()
        RemoteControllerInterface.__init__(self, identifier='SAC')
        
    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:
        return self.agent.act(obs)
               
if __name__ == '__main__':
    controller = RemoteSAC()
    # Play n (None for an infinite amount) games and quit
    client = Client(username='Q-Tips', # Testuser
                    password='AiShaiL9ch',
                    controller=controller, 
                    output_path='/Users/emilbreustedt/Documents/GitHub/RL-Hockey/remote_games', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)