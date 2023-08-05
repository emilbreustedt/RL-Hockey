import numpy as np

from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class RemoteRandomOpponent(RemoteControllerInterface):

    def __init__(self):
        RemoteControllerInterface.__init__(self, identifier='StrongBasicOpponent')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return np.random.uniform(-1,1,4)
        

if __name__ == '__main__':
    controller = RemoteRandomOpponent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Q-Tips', # Testuser
                    password='AiShaiL9ch',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/Q-Tips', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)