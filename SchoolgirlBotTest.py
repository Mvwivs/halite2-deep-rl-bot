
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.stderr = open('errors.log', 'w')

import time
import numpy as np
from itertools import chain
import tensorflow as tf

import hlt

from envs.halite_env import navigate
from supervised_learning.feature_extractor import largest_planet, set_independent_features, set_political_features, get_ship_feature
from supervised_learning.feature_extractor import Ship, Position, Planet

def parse_features(ships_left, ships_right, planets_left, planets_right, planets_empty, center):
    print(f'{ships_left=}\n{ships_right=}\n{planets_left=}\n{planets_right=}\n{planets_empty=}\n', file=sys.stderr)
    features = []

    largest_left = largest_planet(planets_left) if len(planets_left) > 0 else None
    largest_right = largest_planet(planets_right) if len(planets_right) > 0 else None
    largest_empty = largest_planet(planets_empty) if len(planets_empty) > 0 else None

    planet_features = np.zeros((28, 16))

    set_independent_features(chain(planets_left, planets_right, planets_empty), planet_features, center)

    planet_features_left = np.zeros((28, 7))
    planet_features_right = np.zeros((28, 7))

    for planet in chain(planets_left, planets_right, planets_empty):
        set_political_features(planet, left_ships=ships_left, right_ships=ships_right,
                                left_feature=planet_features_left[planet.id],
                                right_feature=planet_features_right[planet.id])

    for ship in (ship for ship in ships_left if not ship.is_docked):
        feature = get_ship_feature(ship=ship,
                                    planets_empty=planets_empty,
                                    enemy_planets=planets_right,
                                    your_planets=planets_left,
                                    largest_empty=largest_empty,
                                    largest_enemy_planet=largest_right,
                                    largest_friendly_planet=largest_left,
                                    common_planet_features=planet_features,
                                    your_planet_features=planet_features_left,
                                    enemy_ships=ships_right,
                                    your_ships=ships_left)
        features.append(feature)

    return features

def get_ships(map: hlt.game_map.Map):

    ships_left = []
    ships_right = []
    for ship in map._all_ships():
        player_id = map.get_me().id
        s = Ship(
            health=ship.health,
            id=ship.id,
            vel_x=ship.vel_x,
            vel_y=ship.vel_y,
            x=ship.x,
            y=ship.y,
            owner=ship.owner.id,
            cooldown=ship._weapon_cooldown,
            is_docked=(ship.docking_status != hlt.entity.Ship.DockingStatus.UNDOCKED)
        )
        if ship.owner.id == player_id:
            ships_left.append(s)
        else:
            ships_right.append(s)

    return ships_left, ships_right

def get_planets(map: hlt.game_map.Map):
    planets_left = []
    planets_right = []
    planets_empty = []

    for planet in map.all_planets():
        p = Planet(
            current_production=planet.current_production,
            health=planet.health,
            id=planet.id,
            owner=planet.owner.id if planet.is_owned() else None,
            remaining_production=planet.remaining_resources,
            x=planet.x,
            y=planet.y,
            docked_ships=len(planet._docked_ships),
            radius=planet.radius
        )
        if p.owner is None:
            planets_empty.append(p)
        elif p.owner == map.get_me().id:
            planets_left.append(p)
        else:
            planets_right.append(p)

    return planets_left, planets_right, planets_empty
            
def get_features(map: hlt.game_map.Map):
    center = Position(x=map.width / 2, y=map.height / 2)
    
    ships_left, ships_right = get_ships(map)
    planets_left, planets_right, planets_empty = get_planets(map)

    return parse_features(ships_left, ships_right, planets_left, planets_right, planets_empty, center)


import hlt

import logging

model = tf.keras.models.load_model('supervised_learning/weights2.h5')
game = hlt.Game("Schoolgirl")

logging.info("Starting my bot!")

while True:
    map: hlt.game_map.Map = game.update_map()
    start_round = time.time()
    player_id = map.get_me().id

    features = np.array(get_features(map))
    # print(f'{features=}\n', file=sys.stderr)
    print(f'{features.shape=}\n', file=sys.stderr)
    if features.shape[0] == 0:
        game.send_command_queue([])
        continue
    actions = model.predict(features)

    command_queue = []
    
    ships_undocked = [ship for ship in map.get_me().all_ships()
                    if ship.docking_status == ship.DockingStatus.UNDOCKED]

    for i, ship in enumerate(ships_undocked):

        navigate_command = None
        for planet in map.all_planets():
            is_planet_friendly = not planet.is_owned() or planet.owner.id == player_id or planet.all_docked_ships() == 0
            if ship.can_dock(planet):
                if is_planet_friendly:
                    if not planet.is_full():
                        navigate_command = ship.dock(planet)
                else:
                    docked_ships = planet.all_docked_ships()
                    weakest_ship = None
                    for s in docked_ships:
                        if weakest_ship is None or weakest_ship.health > s.health:
                            weakest_ship = s
                    navigate_command = navigate(map, start_round, ship,
                            ship.closest_point_to(weakest_ship))
                break

        if navigate_command is None:
            magnitude = actions[i][0] / 2
            if magnitude > 7:
                magnitude = 7
            elif magnitude < 0:
                magnitude = 0
            angle = actions[i][1]
            navigate_command = ship.thrust(magnitude, angle)

        command_queue.append(navigate_command)

    game.send_command_queue(command_queue)

