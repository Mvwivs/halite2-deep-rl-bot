
import numpy as np
import time

import hlt

class MetricsEnv:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        map, reward, done, info = self.env.step(action)
        metrics = self._get_metrics(map)
        info = {**info, **metrics}
        return map, reward, done, info
    
    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def _get_metrics(self, map: hlt.game_map.Map):
        player_id = map.get_me().id
        my_ships = len(map.get_me().all_ships())
        enemy_ships = len(map._all_ships()) - my_ships

        my_planets = 0
        enemy_planets = 0
        for planet in map.all_planets():
            if planet.is_owned():
                if planet.owner.id == player_id:
                    my_planets += 1
                else:
                    enemy_planets += 1

        metrics = {
            'my_ships': my_ships,
            'enemy_ships': enemy_ships,
            'my_planets': my_planets,
            'enemy_planets': enemy_planets,
        }
        return metrics
