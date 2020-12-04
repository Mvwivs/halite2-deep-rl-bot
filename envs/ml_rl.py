
import numpy as np
import time
import heapq

import hlt

from envs.halite_env import navigate
from envs.halite_env import max_planets, max_radius, max_health, max_distance
from envs.gym_discrete import gym_discrete

feature_len = 14

class MlrlEnv():
    def __init__(self, env):
        self.env = env
        self.action_space = gym_discrete(max_planets)
        self.observation_space = np.zeros((max_planets, feature_len))
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

        return self._get_observations(self.map)

    def close(self):
        self.env.close()
    
    def _get_commands(self, action, map: hlt.game_map.Map):
        # print(f'{action=}')
        commands = []
        player_id = map.get_me().id

        total = np.sum(np.exp(action), -1)
        # print(f'{total=}')
        predictions = np.exp(action) / total
        # print(f'{predictions=}')

        ships_to_planets_assignment = self.produce_ships_to_planets_assignment(map, predictions)
        commands = self.produce_instructions(map, ships_to_planets_assignment)

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

        my_ships = len(map.get_me().all_ships())

        if is_docking:
            reward += 10

        if my_planets > self.last_planets:
            reward += 100 * (my_planets - self.last_planets)
        if my_planets < self.last_planets:
            reward -= 100 * (self.last_planets - my_planets)
        self.last_planets = my_planets

        # if my_ships > self.last_ships:
        #     reward += 5 * (my_ships - self.last_ships)
        # if my_ships < self.last_ships:
        #     reward -= 5 * (self.last_ships - my_ships)
        # self.last_ships = my_ships

        return reward

    def _get_observations(self, map):

        player_id = map.get_me().id

        observation = self.observation_space
        for i, planet in enumerate(map.all_planets()):
            if planet is None: # destroyed?
                continue

            exists = True
            owned = -1 # by enemy
            if planet.owner == map.get_me():
                owned = 1
            elif planet.owner is None:
                owned = 0
            
            radius = planet.radius
            health = planet.health
            docked_ships = len(planet.all_docked_ships()) - planet.num_docking_spots
            production = planet.current_production * owned
            remaining_resources = planet.remaining_resources
            
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
                        enemy_gravity -= ship.health / (d * d)

            health_weighted_ship_distance = health_weighted_ship_distance / sum_of_health
            distance_from_center = np.linalg.norm((planet.x - map.width / 2, planet.y - map.height / 2))
            
            is_active = docked_ships > 0 or owned != 1

            features = [
                exists,
                owned,
                radius,
                health,
                production,
                docked_ships,
                friendly_ship_distance,
                enemy_ship_distance,
                is_active,
                friendly_gravity,
                enemy_gravity,
                remaining_resources,
                health_weighted_ship_distance,
                distance_from_center
            ]
            observation[i] = np.array(features)

        return observation

    def produce_ships_to_planets_assignment(self, game_map, predictions):
        """
        Given the predictions from the neural net, create assignment (undocked ship -> planet) deciding which
        planet each ship should go to. Note that we already know how many ships is going to each planet
        (from the neural net), we just don't know which ones.
        :param game_map: game map
        :param predictions: probability distribution describing where the ships should be sent
        :return: list of pairs (ship, planet)
        """
        undocked_ships = [ship for ship in game_map.get_me().all_ships()
                          if ship.docking_status == ship.DockingStatus.UNDOCKED]

        # greedy assignment
        assignment = []
        number_of_ships_to_assign = len(undocked_ships)

        if number_of_ships_to_assign == 0:
            return []

        planet_heap = []
        ship_heaps = [[] for _ in range(max_planets)]

        # Create heaps for greedy ship assignment.
        for planet in game_map.all_planets():
            # We insert negative number of ships as a key, since we want max heap here.
            heapq.heappush(planet_heap, (-predictions[planet.id] * number_of_ships_to_assign, planet.id))
            h = []
            for ship in undocked_ships:
                d = ship.calculate_distance_between(planet)
                heapq.heappush(h, (d, ship.id))
            ship_heaps[planet.id] = h

        # Create greedy assignment
        already_assigned_ships = set()

        while number_of_ships_to_assign > len(already_assigned_ships):
            # Remove the best planet from the heap and put it back in with adjustment.
            # (Account for the fact the distribution values are stored as negative numbers on the heap.)
            ships_to_send, best_planet_id = heapq.heappop(planet_heap)
            ships_to_send = -(-ships_to_send - 1)
            heapq.heappush(planet_heap, (ships_to_send, best_planet_id))

            # Find the closest unused ship to the best planet.
            _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            while best_ship_id in already_assigned_ships:
                _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])

            # Assign the best ship to the best planet.
            assignment.append(
                (game_map.get_me().get_ship(best_ship_id), game_map.get_planet(best_planet_id)))
            already_assigned_ships.add(best_ship_id)

        return assignment

    def produce_instructions(self, game_map, ships_to_planets_assignment):
        """
        Given list of pairs (ship, planet) produce instructions for every ship to go to its respective planet.
        If the planet belongs to the enemy, we go to the weakest docked ship.
        If it's ours or is unoccupied, we try to dock.
        :param game_map: game map
        :param ships_to_planets_assignment: list of tuples (ship, planet)
        :param round_start_time: time (in seconds) between the Epoch and the start of this round
        :return: list of instructions to send to the Halite engine
        """
        command_queue = []
        # Send each ship to its planet
        for ship, planet in ships_to_planets_assignment:

            is_planet_friendly = not planet.is_owned() or planet.owner == game_map.get_me()

            if is_planet_friendly:
                if ship.can_dock(planet):
                    command_queue.append(ship.dock(planet))
                else:
                    command_queue.append(
                        navigate(game_map, self.start_round, ship, ship.closest_point_to(planet)))
            else:
                docked_ships = planet.all_docked_ships()
                assert len(docked_ships) > 0
                weakest_ship = None
                for s in docked_ships:
                    if weakest_ship is None or weakest_ship.health > s.health:
                        weakest_ship = s
                command_queue.append(
                    navigate(game_map, self.start_round, ship, ship.closest_point_to(weakest_ship)))
        return command_queue
