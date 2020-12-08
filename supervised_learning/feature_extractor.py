import argparse
import math
from itertools import chain
from pathlib import Path

import hyperjson as json
from collections import namedtuple

import numpy as np

Ship = namedtuple('Ship', ('health', 'id', 'owner', 'vel_x', 'vel_y', 'x', 'y', 'cooldown', 'is_docked'))
Position = namedtuple('Position', ('x', 'y'))
Planet = namedtuple('Planet',
                    ('current_production', 'radius', 'docked_ships', 'health', 'id', 'owner', 'remaining_production',
                     'x', 'y'))


class Move:
    magnitude = 0
    angle = 1


class Feature:
    cooldown = 0
    health = 1
    x = 2
    y = 3
    vel_x = 4 # !!!
    vel_y = 5 # !!!
    closest_empty_planet_distance = 6
    closest_empty_planet_angle = 7
    distance_to_closest_friendly_planet = 8
    angle_to_closest_friendly_planet = 9
    distance_to_closest_enemy_planet = 10
    angle_to_closest_enemy_planet = 11
    distance_to_largest_friendly_planet = 12
    angle_to_largest_friendly_planet = 13
    distance_to_largest_enemy_planet = 14
    angle_to_largest_enemy_planet = 15
    distance_to_closest_friendly_ship = 16 # !!!
    angle_to_closest_friendly_ship = 17 # !!!
    distance_to_closest_enemy_ship = 18
    angle_to_closest_enemy_ship = 19
    distance_to_largest_empty_planet = 20
    angle_to_largest_empty_planet = 21


class FeaturePlanets:
    owner = 0
    friendly_ship_distance = 1
    enemy_ship_distance = 2
    is_active = 3
    friendly_gravity = 4
    enemy_gravity = 5
    health_weighted_ship_distance = 6

    exists = 7
    radius = 8
    health = 9 # !!!
    production = 10
    docked_ships = 11
    remaining_production = 12 # !!!
    distance_from_center = 13

    distance = 14
    angle = 15


def make_move(move_dict):
    if move_dict is None or move_dict['type'] != 'thrust':
        return None
    move = np.zeros(2)
    move[Move.angle] = move_dict['angle']
    move[Move.magnitude] = move_dict['magnitude']
    return move


def make_ship(ship_dict):
    return Ship(
        health=ship_dict['health'],
        id=ship_dict['id'],
        vel_x=ship_dict['vel_x'],
        vel_y=ship_dict['vel_y'],
        x=ship_dict['x'],
        y=ship_dict['y'],
        owner=ship_dict['owner'],
        cooldown=ship_dict['cooldown'],
        is_docked=ship_dict['docking']['status'] != "undocked"
    )


def make_planet(planet_dict, planets_info):
    planet_id = planet_dict['id']
    return Planet(
        current_production=planet_dict['current_production'],
        health=planet_dict['health'],
        id=planet_id,
        owner=planet_dict['owner'],
        remaining_production=planet_dict['remaining_production'],
        x=planets_info[planet_id]['x'],
        y=planets_info[planet_id]['y'],
        docked_ships=len(planet_dict['docked_ships']),
        radius=planets_info[planet_id]['r']
    )


def parse_planets(planets):
    planets_empty = []
    planets_left = []
    planets_right = []
    for planet in planets:
        if planet.owner is None:
            planets_empty.append(planet)
        elif planet.owner == 0:
            planets_left.append(planet)
        else:
            planets_right.append(planet)
    return planets_left, planets_right, planets_empty


def find_closest_planet_distance(ship, planets):
    if len(planets) == 0:
        return 1000, None

    distances = [
        distance_between(ship, planet)
        for planet in planets
    ]
    return np.min(distances), planets[np.argmin(distances)]


def distance_between(ship, planet):
    return np.linalg.norm((planet.x - ship.x, planet.y - ship.y))


def angle_between(lhs, rhs):
    if rhs is None:
        return 1000
    return math.degrees(math.atan2(lhs.y - rhs.y, rhs.x - lhs.x)) % 360


def largest_planet(planets):
    radius, planet_idx = -1, -1
    for i, planet in enumerate(planets):
        if planet.radius > radius:
            radius = planet.radius
            planet_idx = i
    return planets[planet_idx]


def find_closest_ship(target, ships):
    if len(ships) == 0:
        return 1000, None

    distances = [
        distance_between(target, ship)
        for ship in ships
    ]
    return np.min(distances), ships[np.argmin(distances)]


def set_independent_features(planets, planet_features, center):
    for planet in planets:
        planet_features[planet.id][FeaturePlanets.exists] = 1
        planet_features[planet.id][FeaturePlanets.health] = planet.health
        planet_features[planet.id][FeaturePlanets.radius] = planet.radius

        planet_features[planet.id][FeaturePlanets.production] = planet.current_production
        planet_features[planet.id][FeaturePlanets.docked_ships] = planet.docked_ships
        planet_features[planet.id][FeaturePlanets.remaining_production] = planet.remaining_production
        planet_features[planet.id][FeaturePlanets.distance_from_center] = distance_between(center, planet)


def set_political_features(planet, left_feature, right_feature, left_ships, right_ships):
    if planet.owner == 0:
        left_feature[FeaturePlanets.owner] = 1
        right_feature[FeaturePlanets.owner] = -1
    elif planet.owner == 1:
        right_feature[FeaturePlanets.owner] = 1
        left_feature[FeaturePlanets.owner] = -1
    else:
        left_feature[FeaturePlanets.owner] = 0
        right_feature[FeaturePlanets.owner] = 0

    health_weighted_ship_distance_left = 0
    health_weighted_ship_distance_right = 0
    sum_of_health_left = 0
    sum_of_health_right = 0
    right_gravity = 0
    left_gravity = 0
    right_ship_distance = np.Inf
    left_ship_distance = np.Inf

    for ship in left_ships:
        d = distance_between(ship, planet)
        left_ship_distance = min(left_ship_distance, d)
        sum_of_health_left += ship.health
        health_weighted_ship_distance_left += d * ship.health
        left_gravity += ship.health / (d * d)

    for ship in right_ships:
        d = distance_between(ship, planet)
        right_ship_distance = min(right_ship_distance, d)
        health_weighted_ship_distance_right += d * ship.health
        sum_of_health_right += ship.health
        right_gravity += ship.health / (d * d)

    if len(left_ships) > 0:
        health_weighted_ship_distance_left = health_weighted_ship_distance_left / sum_of_health_left

    if len(right_ships) > 0:
        health_weighted_ship_distance_right = health_weighted_ship_distance_right / sum_of_health_right

    left_feature[FeaturePlanets.is_active] = int(
        planet.owner is None or (planet.docked_ships > 0 and planet.owner == 0) or planet.owner == 1)

    right_feature[FeaturePlanets.is_active] = int(
        planet.owner is None or (planet.docked_ships > 0 and planet.owner == 1) or planet.owner == 0)

    left_feature[FeaturePlanets.enemy_gravity] = right_gravity
    left_feature[FeaturePlanets.friendly_gravity] = left_gravity
    left_feature[FeaturePlanets.friendly_ship_distance] = left_ship_distance
    left_feature[FeaturePlanets.enemy_ship_distance] = right_ship_distance
    left_feature[FeaturePlanets.health_weighted_ship_distance] = health_weighted_ship_distance_left

    right_feature[FeaturePlanets.enemy_gravity] = left_gravity
    right_feature[FeaturePlanets.friendly_gravity] = right_gravity
    right_feature[FeaturePlanets.health_weighted_ship_distance] = health_weighted_ship_distance_right
    right_feature[FeaturePlanets.friendly_ship_distance] = right_ship_distance
    right_feature[FeaturePlanets.enemy_ship_distance] = left_ship_distance


def get_ship_feature(ship, planets_empty, enemy_planets, your_planets, largest_empty, largest_enemy_planet,
                     largest_friendly_planet,
                     your_ships, enemy_ships, common_planet_features, your_planet_features):
    feature = np.zeros(28 * 16 + 22)
    feature[Feature.cooldown] = ship.cooldown
    feature[Feature.health] = ship.health
    feature[Feature.x] = ship.x
    feature[Feature.y] = ship.y
    feature[Feature.vel_x] = ship.vel_x
    feature[Feature.vel_y] = ship.vel_y
    feature[Feature.closest_empty_planet_distance], closest_empty_planet = find_closest_planet_distance(
        ship, planets_empty)
    feature[Feature.distance_to_closest_enemy_planet], closest_enemy_planet = find_closest_planet_distance(
        ship, enemy_planets)
    feature[Feature.distance_to_closest_friendly_planet], closest_team_planet = find_closest_planet_distance(ship,
                                                                                                             your_planets)
    feature[Feature.angle_to_closest_enemy_planet] = angle_between(ship, closest_enemy_planet)
    feature[Feature.angle_to_closest_friendly_planet] = angle_between(ship, closest_team_planet)
    feature[Feature.closest_empty_planet_angle] = angle_between(ship, closest_empty_planet)

    if largest_enemy_planet is not None:
        feature[Feature.distance_to_largest_enemy_planet] = distance_between(ship, largest_enemy_planet)
        feature[Feature.angle_to_largest_enemy_planet] = angle_between(ship, largest_enemy_planet)
    else:
        feature[Feature.distance_to_largest_enemy_planet] = 1000
        feature[Feature.angle_to_largest_enemy_planet] = 1000

    if largest_friendly_planet is not None:
        feature[Feature.distance_to_largest_friendly_planet] = distance_between(ship, largest_friendly_planet)
        feature[Feature.angle_to_largest_friendly_planet] = angle_between(ship, largest_friendly_planet)
    else:
        feature[Feature.distance_to_largest_friendly_planet] = 1000
        feature[Feature.angle_to_largest_friendly_planet] = 1000

    if largest_empty is not None:
        feature[Feature.distance_to_largest_empty_planet] = distance_between(ship, largest_empty)
        feature[Feature.angle_to_largest_empty_planet] = angle_between(ship, largest_empty)
    else:
        feature[Feature.distance_to_largest_empty_planet] = 1000
        feature[Feature.angle_to_largest_empty_planet] = 1000

    feature[Feature.distance_to_closest_friendly_ship], closest_team_ship = find_closest_ship(ship, your_ships)
    feature[Feature.distance_to_closest_enemy_ship], closest_enemy_ship = find_closest_ship(ship, enemy_ships)

    feature[Feature.angle_to_closest_friendly_ship] = angle_between(ship, closest_team_ship)
    feature[Feature.angle_to_closest_enemy_ship] = angle_between(ship, closest_enemy_ship)

    # common = np.copy(common_planet_features)
    # common[0:28,0:7] = your_planet_features
    # feature[22:] = common.flatten()
    for planet in chain(your_planets, enemy_planets, planets_empty):
        offset = 22 + planet.id * 16
        feature[offset:offset + 16] = common_planet_features[planet.id]
        feature[offset:offset + 7] = your_planet_features[planet.id]
        feature[offset + FeaturePlanets.distance] = distance_between(ship, planet)
        feature[offset + FeaturePlanets.angle] = angle_between(ship, planet)
    return feature


def parse_replay(path):
    with open(path, "r") as f:
        replay = json.loads(f.readline()[2:-1])
        planets_info = replay['planets']
        frames = replay['frames']
        width, height = replay['width'], replay['height']
        center = Position(x=width / 2, y=height / 2)
        move_frames = [move_frame for move_frame in replay['moves']]
        features = []
        outputs = []
        for move_frame, frame in zip(move_frames, frames):

            planets_left, planets_right, planets_empty = parse_planets(
                make_planet(planet, planets_info)
                for planet in frame['planets'].values()
            )
            ships_raw = frame['ships']
            ships_left = [make_ship(ship) for ship in ships_raw["0"].values()]
            ships_right = [make_ship(ship) for ship in ships_raw["1"].values()]

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

            left_moves = move_frame['0'][0]
            right_moves = move_frame['1'][0]
            for ship in (ship for ship in ships_left if not ship.is_docked):
                expected_output = make_move(left_moves.get(str(ship.id)))
                if expected_output is None:
                    continue
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
                outputs.append(expected_output)

            for ship in (ship for ship in ships_right if not ship.is_docked):
                expected_output = make_move(right_moves.get(str(ship.id)))
                if expected_output is None:
                    continue
                feature = get_ship_feature(ship=ship,
                                           planets_empty=planets_empty,
                                           enemy_planets=planets_right,
                                           your_planets=planets_right,
                                           largest_empty=largest_empty,
                                           largest_enemy_planet=largest_left,
                                           largest_friendly_planet=largest_right,
                                           common_planet_features=planet_features,
                                           your_planet_features=planet_features_right,
                                           enemy_ships=ships_left,
                                           your_ships=ships_right)
                features.append(feature)
                outputs.append(expected_output)

        return features, outputs


def write_features(feature_file, output_file, features, outputs):
    np.save(feature_file, np.array(features), allow_pickle=False)
    np.save(output_file, np.array(outputs), allow_pickle=False)


def extract_features_to_dir(root_dir, features_dir_path):
    for replay in Path(root_dir).iterdir():
        print(replay.absolute())
        target_basename = replay.stem
        features_filename = Path(features_dir_path).joinpath(target_basename + "_features").with_suffix(".npy")
        output_filename = Path(features_dir_path).joinpath(target_basename + "_outputs").with_suffix(".npy")
        f, o = parse_replay(replay.absolute())
        write_features(features_filename, output_filename, f, o)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays")
    parser.add_argument("--features")
    args = parser.parse_args()

    extract_features_to_dir(args.replays, args.features)
