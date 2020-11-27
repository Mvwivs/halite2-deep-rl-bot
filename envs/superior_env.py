import numpy as np
import time
from envs.grid import Grid
from gym.spaces import Discrete
from math import sqrt, floor

import hlt
from halite_env import navigate
from hlt.game_map import Map


class SuperiorEnv:
    def __init__(self, env, squares_num, feature_len):
        self.env = env
        self.tiles_num = squares_num
        self.action_space = Discrete(squares_num)
        self.observation_space = np.zeros((squares_num, feature_len))
        self.map = None
        self.grid = None
        self.start_round = 0
        self.FEATURE_LEN = feature_len

    def step(self, action):
        commands = self._get_commands(action, self.map)
        self.map, _, done, _ = self.env.step(commands)
        self.start_round = time.time()

        reward = self._calc_reward(self.map)
        observation = self._get_observations(self.map)

        return observation, reward, done, {}

    def reset(self):
        self.map: Map = self.env.reset()
        self.grid: Grid = Grid(self.tiles_num, self.map.height, self.map.width)

        return self._get_observations(self.map)

    def close(self):
        self.env.close()

    def configure(self, socket_path="/dev/shm/bot.sock", replay=False):
        self.env.configure(socket_path, replay)

    # go to free planet
    # go to enemy planet
    # go to circle center
    def _get_commands(self, action, world_map: hlt.game_map.Map):
        commands = []
        player_id = world_map.get_me().id
        planets = world_map.all_planets()
        ships = world_map.get_me().all_ships()

        free_planets = []
        enemy_planets = []
        target_tile = floor(action)
        for planet in planets:
            planet_tile = self.grid.get_tile_id(planet.x, planet.y)
            if planet_tile == target_tile and not planet.is_owned():
                free_planets.append(planet)
            elif planet_tile == target_tile and planet.owner.id != player_id:
                enemy_planets.append(planet)

        if len(free_planets) > 0:
            for ship in ships:
                dest_planet = free_planets[np.argmin([
                    np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                    for planet in free_planets
                ])]
                if ship.can_dock(dest_planet):
                    ship.dock(dest_planet)
                else:
                    commands.append(navigate(
                        world_map,
                        self.start_round,
                        ship,
                        ship.closest_point_to(dest_planet),
                        speed=int(hlt.constants.MAX_SPEED / 2)))
        elif len(enemy_planets) > 0:
            for ship in ships:
                dest_planet = enemy_planets[np.argmin([
                    np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                    for planet in enemy_planets
                ])]
                if ship.can_dock(dest_planet):
                    ship.dock(dest_planet)
                else:
                    commands.append(navigate(
                        world_map,
                        self.start_round,
                        ship,
                        ship.closest_point_to(dest_planet),
                        speed=int(hlt.constants.MAX_SPEED / 2)))
        else:
            tile_center = self.grid.get_tile_center_by_id(target_tile)
            for ship in ships:
                commands.append(navigate(
                    world_map,
                    self.start_round,
                    ship,
                    tile_center,
                    speed=int(hlt.constants.MAX_SPEED / 2)))

        return commands

    def _calc_reward(self, map: hlt.game_map.Map):
        player_id = map.get_me().id
        my_planets = 0
        for planet in map.all_planets():
            if planet.is_owned() and planet.owner.id == player_id:
                my_planets += 1
        return my_planets / len(map.all_planets())

    # Общее здоровье планет в тайле / общее здоровье
    # Общее количество союзных пришвартованных кораблей /  максимальное количество пришвартованных кораблей в этом
    # кводрате
    # Общее количество враждебных пришвартованных кораблей / максимальное количество пришвартованных кораблей в этом
    # кводрате
    # Расстояние от центра квадрата до ближайшего союзного корабля / диагональ
    # Расстояние от центра квадрата до ближайшего вражеского корабля / диагональ
    # Количество противников / общее число кораблей в игре
    # Количество союзников / общее число кораблей в игре
    # Количество зохваченыхъъъ планет  / общее число планет
    # Количество незохваченыхъъъ планет / общее число планет
    # Количество планет / общее число планет

    def _get_observations(self, world_map: Map):

        player_id = world_map.get_me().id
        planets = world_map.all_planets()
        all_ships = world_map._all_ships()

        observation = np.zeros((self.tiles_num, self.FEATURE_LEN))

        total_planets_health = 0
        total_docking_spots = np.zeros(self.tiles_num)
        diagonal = sqrt(self.map.height ** 2 + self.map.width ** 2)
        total_ships_number = len(all_ships)
        total_planets = len(planets)

        my_distances = np.zeros(self.tiles_num)
        enemy_distances = np.zeros(self.tiles_num)

        for ship in all_ships:
            ship_docked = ship.planet is not None
            ship_mine = ship.owner == player_id
            if ship_docked and ship_mine:
                enemy_docked, my_docked = 0, int(ship_mine)
            elif ship_docked:
                enemy_docked, my_docked = int(ship_mine), 0
            else:
                enemy_docked, my_docked = 0, 0

            features = [
                0,
                my_docked,
                enemy_docked,
                0,
                0,
                int(ship_mine),
                int(not ship_mine),
                0,
                0,
                0,
            ]
            observation[self.grid.get_tile_id(ship.x, ship.y)] += np.array(features)
            for x in range(self.grid.grid_width):
                for y in range(self.grid.grid_height):
                    center_x, center_y = self.grid.get_tile_center(x, y)
                    distance_to_ship = np.linalg.norm((center_x - ship.x, center_y - ship.y))
                    tile_id = self.grid.get_tile_id(center_x, center_y)
                    if ship_mine:
                        min_distance_to_tile = my_distances[tile_id]
                        if min_distance_to_tile > distance_to_ship:
                            my_distances[tile_id] = distance_to_ship
                    else:
                        min_distance_to_tile = enemy_distances[tile_id]
                        if min_distance_to_tile > distance_to_ship:
                            enemy_distances[tile_id] = distance_to_ship

        for i, planet in enumerate(world_map.all_planets()):
            tile_id = self.grid.get_tile_id(planet.x, planet.y)
            docked_ships = planet.all_docked_ships()

            total_docking_spots[tile_id] += planet.num_docking_spots
            if planet.is_owned() and planet.owner.id == player_id:
                enemy_docked, my_docked = 0, len(docked_ships)
            elif planet.is_owned():
                enemy_docked, my_docked = len(docked_ships), 0
            else:
                enemy_docked, my_docked = 0, 0

            total_planets_health += planet.health
            owned = planet.is_owned()

            features = [
                planet.health,
                my_docked,
                enemy_docked,
                0,
                0,
                0,
                0,
                int(owned),
                int(not owned),
                1
            ]
            observation[tile_id] += np.array(features)

        for i in range(self.tiles_num):
            tile_features = observation[i]
            tile_features[0] /= total_planets_health
            tile_features[1] /= total_docking_spots[i]
            tile_features[2] /= total_docking_spots[2]
            tile_features[3] = my_distances[i] / diagonal
            tile_features[4] = enemy_distances[i] / diagonal
            tile_features[5] /= total_ships_number
            tile_features[6] /= total_ships_number
            tile_features[7] /= total_planets
            tile_features[8] /= total_planets
            tile_features[9] /= total_planets

        return observation
