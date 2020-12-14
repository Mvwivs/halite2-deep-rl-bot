
import numpy as np
import time
import heapq

import gym

import hlt

from envs.halite_env import navigate
from envs.halite_env import max_planets, max_radius, max_health, max_distance
from envs.gym_discrete import gym_discrete

max_ships = 300 # ?
max_production = 10000 # ?
max_ship_health = 255

class AttractionEnv():
    def __init__(self, env):
        self.env = env
        self.feature_len = 15
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(max_planets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(max_planets, self.feature_len), dtype=np.float32)
        self.metadata = None
        self.reward_range = (-float('inf'), float('inf'))
        self.map = None
        self.start_round = 0

        self.last_planets = 0
        self.last_ships = 0


    def step(self, action):
        
        commands = self._get_commands(action, self.map)
        self.map, _, done, info = self.env.step(commands)
        self.start_round = time.time()

        reward = self._calc_reward(self.map)
        observation = self._get_observations(self.map)

        return observation, reward, done, info
    
    def reset(self):
        self.map = self.env.reset()

        self.start_round = 0

        self.last_planets = 0
        self.last_ships = 0

        return self._get_observations(self.map)

    def close(self):
        self.env.close()
    
    def _get_commands(self, actions, map: hlt.game_map.Map):
        action_ = actions.reshape((1, max_planets))[0]
        # print(f'{actions=}')
        commands = []
        player_id = map.get_me().id

        unshuffled = {}
        for id, pos in self.shuffle_index.items():
            if id != -1:
                unshuffled[id] = action_[pos]

        undocked_ships = [ship for ship in map.get_me().all_ships()
                    if ship.docking_status == ship.DockingStatus.UNDOCKED]

        for ship in undocked_ships:
            max_gravity = -np.Inf
            max_planet = None
            for planet_id, attraction in unshuffled.items():
                planet = map.get_planet(planet_id)
                dist = ship.calculate_distance_between(planet)
                gravity = 1.1 * (attraction ** 3) - dist / max_distance
                if gravity > max_gravity:
                    max_gravity = gravity
                    max_planet = planet

            is_planet_friendly = not max_planet.is_owned() \
                or max_planet.owner.id == player_id \
                or max_planet.all_docked_ships() == 0
            if is_planet_friendly:
                if ship.can_dock(max_planet):
                        commands.append(ship.dock(max_planet))
                else:
                    navigate_command = navigate(map, self.start_round, ship,
                        ship.closest_point_to(max_planet))
                    if navigate_command:
                        commands.append(navigate_command)
            else:
                docked_ships = max_planet.all_docked_ships()
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
        
        docking_now = 0
        for ship in map.get_me().all_ships():
            if ship.docking_status == hlt.entity.Ship.DockingStatus.DOCKING:
                docking_now += 1
                break

        my_ships = len(map.get_me().all_ships())

        reward += docking_now * 10

        if my_planets > self.last_planets:
            reward += 100 * (my_planets - self.last_planets)
        if my_planets < self.last_planets:
            reward -= 100 * (self.last_planets - my_planets)
        self.last_planets = my_planets

        # if my_ships > self.last_ships:
        #     reward += 5 * (my_ships - self.last_ships)
        if my_ships < self.last_ships:
            reward -= 20 * (self.last_ships - my_ships)
        self.last_ships = my_ships

        return reward

    def _get_observations(self, map):
        player_id = map.get_me().id

        unshuffled = []
        planet: hlt.entity.Planet
        for i, planet in enumerate(map.all_planets()):
            if planet is None: # destroyed?
                continue

            exists = True
            owner_none = 0
            owner_me = 0
            owner_enemy = 0
            if not planet.is_owned():
                owner_none = 1
            elif planet.owner.id == player_id:
                owner_me = 1
            else:
                owner_enemy = 1
            
            radius = planet.radius / max_radius
            health = planet.health / max_health
            docked_ships = len(planet.all_docked_ships()) / planet.num_docking_spots
            production = planet.current_production / (planet.num_docking_spots * 6)
            remaining_resources = planet.remaining_resources / max_production
            
            friendly_ship_distance = max_distance
            enemy_ship_distance = max_distance
            health_weighted_ship_distance = 0
            sum_of_health = 0
            enemy_gravity = 0
            friendly_gravity = 0
            for player in map.all_players():
                for ship in player.all_ships():
                    d = ship.calculate_distance_between(planet)
                    if player == map.get_me():
                        friendly_ship_distance = min(friendly_ship_distance, d)
                        sum_of_health += ship.health
                        health_weighted_ship_distance += d * ship.health
                        friendly_gravity += ship.health / (d * d)
                    else:
                        enemy_ship_distance = min(enemy_ship_distance, d)
                        enemy_gravity += ship.health / (d * d)
            
            friendly_ship_distance = friendly_ship_distance / max_distance
            enemy_ship_distance = enemy_ship_distance/ max_distance

            health_weighted_ship_distance = health_weighted_ship_distance / sum_of_health
            health_weighted_ship_distance = health_weighted_ship_distance / (max_ships * max_distance * max_ship_health)
            distance_from_center = np.linalg.norm((planet.x - map.width / 2, planet.y - map.height / 2)) / max_distance

            friendly_gravity = friendly_gravity / (max_ships * max_ship_health)
            enemy_gravity = enemy_gravity / (max_ships * max_ship_health)
            
            features = [
                exists,
                owner_none,
                owner_me,
                owner_enemy,

                radius,
                health,
                production,
                docked_ships,
                remaining_resources,
                distance_from_center,

                friendly_ship_distance,
                enemy_ship_distance,
                friendly_gravity,
                enemy_gravity,
                health_weighted_ship_distance
            ]
            unshuffled.append((planet.id, np.array(features)))

        # print(f'{unshuffled=}')
        left = max_planets - len(unshuffled)
        for i in range(left, max_planets):
            unshuffled.append((-1, list(np.zeros(self.feature_len))))

        # np.random.shuffle(unshuffled)
        self.shuffle_index = { u[0]:pos for pos, u in enumerate(unshuffled) if u[0] != -1 }

        observation = np.zeros(self.observation_space.shape)
        for i in range(0, len(unshuffled)):
            observation[i] = unshuffled[i][1]

        # print(f'{observation=}')
        return observation
