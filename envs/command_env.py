
import numpy as np
import time

import hlt

from envs.halite_env import navigate
from envs.halite_env import max_planets, max_radius, max_health, max_distance
from envs.gym_discrete import gym_discrete

feature_len = 7

class CommandEnv():
    def __init__(self, env):
        self.min_planet = 12

        self.env = env
        self.action_space = gym_discrete(self.min_planet)
        self.observation_space = np.zeros((self.min_planet, feature_len))
        self.map = None
        self.start_round = 0

        self.last_planets = 0


    def step(self, action):
        
        commands = self._get_commands(action, self.map)
        self.map, _, done, _ = self.env.step(commands)
        self.start_round = time.time()

        reward = self._calc_reward(self.map)
        observation = self._get_observations(self.map)

        return observation, reward, done, {}
    
    def reset(self):
        self.map = self.env.reset()

        sorted_planets = sorted(self.map.all_planets(), key=lambda p: p.radius, reverse=True)[0:self.min_planet]

        self.planet_index = [p.id for p in sorted_planets]
        return self._get_observations(self.map)

    def close(self):
        self.env.close()
    
    def _get_commands(self, action, map: hlt.game_map.Map):
        # print(f'{action=}')
        commands = []
        planet_id_sorted = action
        planet_id = self.planet_index[planet_id_sorted]

        dest_planet = map.get_planet(planet_id)
        if dest_planet is None:
            return []

        player_id = map.get_me().id
        is_planet_friendly = not dest_planet.is_owned() \
            or dest_planet.owner.id == player_id \
            or dest_planet.all_docked_ships() == 0

        for ship in map.get_me().all_ships():
            if is_planet_friendly:
                if ship.can_dock(dest_planet):
                        commands.append(ship.dock(dest_planet))
                else:
                    navigate_command = navigate(map, self.start_round, ship,
                        ship.closest_point_to(dest_planet))
                    if navigate_command:
                        commands.append(navigate_command)
            else:
                docked_ships = dest_planet.all_docked_ships()
                weakest_ship = None
                for s in docked_ships:
                    if weakest_ship is None or weakest_ship.health > s.health:
                        weakest_ship = s
                commands.append(
                    navigate(map, self.start_round, ship,
                        ship.closest_point_to(weakest_ship)))

        return commands

    def _calc_reward(self, map: hlt.game_map.Map):
        reward = 0
        player_id = map.get_me().id

        my_planets = 0
        for planet in map.all_planets():
            if planet.is_owned() and planet.owner.id == player_id:
                my_planets += 1
        
        is_docking = False
        for ship in map.get_me().all_ships():
            if ship.docking_status == hlt.entity.Ship.DockingStatus.DOCKING:
                is_docking = True
                break

        if is_docking:
            reward += 10

        if my_planets > self.last_planets:
            reward += 100

        self.last_planets = my_planets
        return reward

    def _get_observations(self, map):

        player_id = map.get_me().id

        observation = np.zeros((self.min_planet, feature_len))
        for i, planet_id in enumerate(self.planet_index):
            planet = map.get_planet(planet_id)
            if planet is None:
                continue
            
            radius = planet.radius / max_radius
            health = planet.health / max_health
            docked_ships = len(planet.all_docked_ships()) / planet.num_docking_spots
            
            closest_friendly_ship_distance = np.min([ 
                np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                for ship in map.get_me().all_ships()
            ]) / max_distance
            closest_enemy_ship_distance = np.min([ 
                np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                for ship in map._all_ships()
                if ship.owner != player_id
            ]) / max_distance

            owned = planet.is_owned()
            owned_by_me = planet.is_owned() and (planet.owner.id == player_id)

            features = [
                radius,
                health,
                docked_ships,
                closest_friendly_ship_distance,
                closest_enemy_ship_distance,
                owned,
                owned_by_me
            ]
            observation[i] = np.array(features)

        return observation
