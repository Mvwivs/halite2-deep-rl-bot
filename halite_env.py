
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

        observation = self.game.update_map()
        reward = 1

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

class CommandEnv():
    def __init__(self):
        self.env = Env()

    def step(self, actions):
        
        commands = self._get_commands(actions)
        map, _, done, _ = self.env.step(commands)

        reward = self._calc_reward(map)
        observation = self._get_observations(map)

        return observation, reward, done, {}
    
    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
        
    def configure(self, socket_path="/dev/shm/bot.sock"):
        self.env.configure(socket_path)

    def _get_commands(self, actions):
        commands = []
        return commands

    def _calc_reward(self, map):
        reward = 0
        return reward

    def _get_observations(self, map):
        observation = {}
        return observation
