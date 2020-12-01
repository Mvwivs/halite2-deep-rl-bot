import subprocess as sub
import time

import numpy as np
import math

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
            self.process = sub.Popen(
                ["./halite", "-t", "-i", "replays", f'python3 FakeBot.py {self.socket_path}', "python3 Enemy.py"])
        else:
            self.process = sub.Popen(
                ["./halite", "-t", "-q", "-r", "-i", "replays", f'python3 FakeBot.py {self.socket_path}',
                 "python3 Enemy.py"],
                stdout=sub.PIPE)

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

def navigate(game_map, start_of_round, ship, destination, speed=int(hlt.constants.MAX_SPEED/2)):
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
        navigate_command = ship.navigate(destination, game_map, speed=speed, max_corrections=180, ignore_ships=False)
    if navigate_command is None:
        # ship.navigate may return None if it cannot find a path. In such a case we just thrust.
        dist = ship.calculate_distance_between(destination)
        speed = speed if (dist >= speed) else dist
        navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
    return navigate_command


def attack(game_map, start_of_round, ship, destination, speed):
    current_time = time.time()
    have_time = current_time - start_of_round < 1.2
    dist = ship.calculate_distance_between(destination)
    navigate_command = None
    if have_time and dist > hlt.constants.MAX_SPEED:
        navigate_command = ship.navigate(destination, game_map, speed=speed, max_corrections=180, ignore_ships=False)
    elif dist <= speed:
        navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
    if navigate_command is None:
        speed = speed if (dist >= speed) else dist
        navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
    return navigate_command


max_planets = 28
max_radius = 16
max_health = max_radius * 255
max_distance = 462
