import time
from math import sqrt, floor

import numpy as np
from gym.spaces import Discrete

import hlt
from envs.grid import Grid
from halite_env import navigate
from hlt.entity import Position
from hlt.game_map import Map


class SuperiorEnv:
    def __init__(self, env, tiles_num):
        self.env = env
        self.tiles_num = tiles_num
        self.FEATURE_LEN = 10
        self.action_space = Discrete(tiles_num)
        self.observation_space = np.zeros((tiles_num, self.FEATURE_LEN))
        self.map = None
        self.grid = None
        self.start_round = 0

    def step(self, action):
        # print(f'{action=}')
        commands = self._get_commands(action, self.map)
        self.map, _, done, _ = self.env.step(commands)
        self.start_round = time.time()

        reward = self._calc_reward(self.map)
        observation = self._get_observations(self.map)

        return observation, reward, done, {}

    def reset(self):
        self.map: Map = self.env.reset()
        self.grid: Grid = Grid(self.tiles_num, self.map.height, self.map.width)
        # print(self.grid)
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
        all_ships = world_map._all_ships()
        ships = [ship for ship in all_ships if ship.owner.id == player_id]
        enemy_ships = [ship for ship in world_map._all_ships() if ship.owner.id != player_id]

        free_planets = []
        enemy_planets = []
        enemy_ships_on_tile = []
        target_tile = floor(action)
        for planet in planets:
            planet_tile = self.grid.get_tile_id(planet.x, planet.y)
            if planet_tile == target_tile and not planet.is_owned():
                free_planets.append(planet)
            elif planet_tile == target_tile and planet.owner.id != player_id:
                enemy_planets.append(planet)

        tile_center_x, tile_center_y = self.grid.get_tile_center_by_id(target_tile)

        for ship in enemy_ships:
            ship_tile = self.grid.get_tile_id(ship.x, ship.y)
            if ship_tile == target_tile:
                enemy_ships_on_tile.append(ship)

        if len(free_planets) > 0:
            for ship in ships:
                dest_planet = free_planets[np.argmin([
                    np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                    for planet in free_planets
                ])]
                if ship.can_dock(dest_planet):
                    commands.append(ship.dock(dest_planet))
                else:
                    commands.append(navigate(
                        world_map,
                        self.start_round,
                        ship,
                        ship.closest_point_to(dest_planet),
                        speed=int(hlt.constants.MAX_SPEED / 2)))
        # elif len(enemy_ships_on_tile) > 0:
        #     for ship in ships:
        #         dest_ship = enemy_ships_on_tile[np.argmin([
        #             np.linalg.norm((enemy.x - ship.x, enemy.y - ship.y))
        #             for enemy in enemy_ships_on_tile
        #         ])]
        #         commands.append(attack(
        #             world_map,
        #             ship,
        #             Position(dest_ship.x, dest_ship.y),
        #             speed=int(hlt.constants.MAX_SPEED / 2)))
        elif len(enemy_planets) > 0:
            for ship in ships:
                dest_planet = enemy_planets[np.argmin([
                    np.linalg.norm((planet.x - ship.x, planet.y - ship.y))
                    for planet in enemy_planets
                ])]
                if len(dest_planet.all_docked_ships()) == 0:
                    if ship.can_dock(dest_planet):
                        commands.append(ship.dock(dest_planet))
                    else:
                        commands.append(navigate(
                            world_map,
                            self.start_round,
                            ship,
                            ship.closest_point_to(dest_planet),
                            speed=int(hlt.constants.MAX_SPEED / 2)))
                else:
                    weakest_ship = None
                    for s in dest_planet.all_docked_ships():
                        if weakest_ship is None or weakest_ship.health > s.health:
                            weakest_ship = s
                    commands.append(
                        navigate(world_map, self.start_round, ship, ship.closest_point_to(weakest_ship),
                                 int(hlt.constants.MAX_SPEED / 2)))

        else:
            for ship in ships:
                commands.append(navigate(
                    world_map,
                    self.start_round,
                    ship,
                    Position(tile_center_x, tile_center_y),
                    speed=int(hlt.constants.MAX_SPEED / 2)))

        return commands

    def _calc_reward(self, map: hlt.game_map.Map):
        player_id = map.get_me().id
        planets = map.all_planets()
        my_planets = 0
        total_planets = len(planets)
        for planet in planets:
            if planet.is_owned() and planet.owner.id == player_id:
                my_planets += 1
        return (my_planets / total_planets) ** 2

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

        my_distances = np.full(self.tiles_num, np.inf)
        enemy_distances = np.full(self.tiles_num, np.inf)

        for ship in all_ships:
            ship_mine = ship.owner.id == player_id
            features = [
                0,
                0,
                0,
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
            docking_spots = total_docking_spots[i]
            if docking_spots != 0:
                tile_features[1] /= total_docking_spots[i]
                tile_features[2] /= total_docking_spots[i]
            tile_features[3] = my_distances[i] / diagonal
            tile_features[4] = enemy_distances[i] / diagonal
            tile_features[5] /= total_ships_number
            tile_features[6] /= total_ships_number
            tile_features[7] /= total_planets
            tile_features[8] /= total_planets
            tile_features[9] /= total_planets
        # print(f'{observation=}')
        return observation
