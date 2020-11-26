
import time
import subprocess as sub
import numpy as np

import hlt

class Env():
    def __init__(self):
        self.game = None
        self.process = None
        self.socket_path = ""
        self.replay = False

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
        if self.replay:
            self.process = sub.Popen(["./halite", "-i", "replays", "-d", "240 160", f'python3 FakeBot.py {self.socket_path}', "python3 Enemy.py"])
        else:
            self.process = sub.Popen(["./halite", "-r", "-i", "replays", "-d", "240 160", f'python3 FakeBot.py {self.socket_path}', "python3 Enemy.py"])
        
        self.game = hlt.GameUnix("Env", self.socket_path)

        return self.game.update_map()

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
            
        if self.process is not None:
            self.process.wait()
            self.process = None
        
    def configure(self, socket_path="/dev/shm/bot.sock", replay=False):
        self.socket_path = socket_path
        self.replay = replay

def navigate(game_map, start_of_round, ship, destination, speed):
    """
    Send a ship to its destination. Because "navigate" method in Halite API is expensive, we use that method only if
    we haven't used too much time yet.

    :param game_map: game map
    :param start_of_round: time (in seconds) between the Epoch and the start of this round
    :param ship: ship we want to send
    :param destination: destination to which we want to send the ship to
    :param speed: speed with which we would like to send the ship to its destination
    :return:
    """
    current_time = time.time()
    have_time = current_time - start_of_round < 1.2
    navigate_command = None
    if have_time:
        navigate_command = ship.navigate(destination, game_map, speed=speed, max_corrections=180)
    if navigate_command is None:
        # ship.navigate may return None if it cannot find a path. In such a case we just thrust.
        dist = ship.calculate_distance_between(destination)
        speed = speed if (dist >= speed) else dist
        navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
    return navigate_command

class CommandEnv():
    def __init__(self):
        self.env = Env()
        self.map = None
        self.start_round = 0

    def step(self, action):
        
        commands = self._get_commands(action, self.map)
        self.map, _, done, _ = self.env.step(commands)
        self.start_round = time.time()

        reward = self._calc_reward(self.map)
        observation = self._get_observations(self.map)

        return observation, reward, done, {}
    
    def reset(self):
        self.map = self.env.reset()
        return self._get_observations(self.map)

    def close(self):
        self.env.close()
        
    def configure(self, socket_path="/dev/shm/bot.sock", replay=False):
        self.env.configure(socket_path, replay)
    
    def _get_commands(self, action, map: hlt.game_map.Map):
        # print(f'{action=}')
        commands = []
        planet_id = action
        if planet_id >= len(map.all_planets()):
            return []

        dest_planet = map.get_planet(planet_id)
        for ship in map.get_me().all_ships():
            if ship.can_dock(dest_planet):
                    commands.append(ship.dock(dest_planet))
            else:
                navigate_command = navigate(
                    map,
                    self.start_round,
                    ship,
                    ship.closest_point_to(dest_planet),
                    speed=int(hlt.constants.MAX_SPEED/2))
                if navigate_command:
                    commands.append(navigate_command)

        return commands

    def _calc_reward(self, map: hlt.game_map.Map):
        player_id = map.get_me().id
        my_planets = 0
        for planet in map.all_planets():
            if planet.is_owned() and planet.owner.id == player_id:
                my_planets += 1
        planet_reward = my_planets / len(map.all_planets())

        my_ships = len(map.get_me().all_ships())
        ship_reward = my_ships / len(map._all_ships())
        # print(f'{ship_reward=}, {planet_reward=}')

        reward = planet_reward
        return reward

    def _get_observations(self, map):
        max_planets = 28
        max_radius = 16
        max_health = max_radius * 255
        feature_len = 5
        max_distance = 462

        player_id = map.get_me().id

        observation = np.zeros((max_planets, feature_len))
        for i, planet in enumerate(map.all_planets()):
            
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

            features = [
                radius,
                health,
                docked_ships,
                closest_friendly_ship_distance,
                closest_enemy_ship_distance,
            ]
            observation[i] = np.array(features)

        return observation
