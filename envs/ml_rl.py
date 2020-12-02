
import numpy as np
import time
import heapq

import hlt

from gym.spaces.discrete import Discrete

from halite_env import navigate
from halite_env import max_planets, max_radius, max_health, max_distance

feature_len = 8

class MlrlEnv():
    def __init__(self, env):
        self.env = env
        self.action_space = Discrete(max_planets)
        self.observation_space = np.zeros((max_planets, feature_len))
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

        self.start_round = 0

        self.last_planets = 0

        return self._get_observations(self.map)

    def close(self):
        self.env.close()
    
    def _get_commands(self, action, map: hlt.game_map.Map):
        # print(f'{action=}')
        commands = []
        player_id = map.get_me().id
        predictions = action

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

        if is_docking:
            reward += 10

        if my_planets > self.last_planets:
            reward += 100

        self.last_planets = my_planets
        return reward

    def _get_observations(self, map):

        player_id = map.get_me().id

        observation = self.observation_space
        for i, planet in enumerate(map.all_planets()):
            if planet is None: # destroyed?
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

            exists = True
            owned = planet.is_owned()
            owned_by_me = planet.is_owned() and (planet.owner.id == player_id)

            features = [
                exists,
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
