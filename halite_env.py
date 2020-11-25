
import time
import subprocess as sub

import hlt

class Env():
    def __init__(self):
        self.game = None
        self.process = None
        self.socket_path = ""

    def step(self, actions):
        self.game.send_command_queue(actions)

        reward = 14_88

        observation = self.game.update_map()
        done = self.game.done
        if done:
            self.close()

        return observation, reward, done, {}
    
    def reset(self):
        self.close()

        # run exe
        self.process = sub.Popen(["./halite", "-r", "-d", "240 160", f'python3 FakeBot.py {self.socket_path}', "python3 Enemy.py"])
        
        self.game = hlt.GameUnix("Env", self.socket_path)

        return self.game.update_map()

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
            
        if self.process is not None:
            self.process.wait()
            self.process = None
        
    def configure(self, socket_path="/dev/shm/bot.sock"):
        self.socket_path = socket_path
