
import hlt
import logging
from collections import OrderedDict


def entity_list_sort_by_distance(_entity):
    """
    @fn     entity_list_sort_by_distance 
    
    @brief  get list of all entities sorted by distance
    
    @param  Dict of All entity
    
    @return OrderDict of entity
    """
    list_sort_by_distance = OrderedDict(
        sorted(_entity.items(), key=lambda t: t[0]))
    return list_sort_by_distance

def get_all_closest_empty_planets_sorted_by_distance(_entity):
    """
    @fn     get_all_closest_empty_planets_sorted_by_distance 
    
    @brief  get list of all empty closest planet
    
    @param  Dict of All entity
    
    @return List of closest empty planet
    """
    closest_empty_planets = []
    for distance in _entity:
        entity_planet = _entity[distance][0]
        if isinstance(entity_planet, hlt.entity.Planet) and not entity_planet.is_owned():
            closest_empty_planets.append(entity_planet)
    return closest_empty_planets


def get_all_enemy_ships(_entity: "OrderDict", _allies_ships: "List"):
    """
    @fn     get_all_enemy_ships 
    
    @brief  get list of all enemy ships
    
    @param  OrderDict of All entity, List of allies ships
    
    @return List of enemy ships
    """
    enemy_ships = []
    for distance in _entity:
        entity_enemy_ship = _entity[distance][0]
        if isinstance(entity_enemy_ship, hlt.entity.Ship) and entity_enemy_ship not in _allies_ships:
            enemy_ships.append(entity_enemy_ship)
    return enemy_ships


game = hlt.Game("Covid-2019")
logging.info("Starting Covid-2019")
planned_planets = []
# logging.info(f'== : {}')
while True:
    game_map = game.update_map()
    command_queue = []
    allies_ships = game_map.get_me().all_ships()

    for ship in allies_ships:
        this_ship_id = ship.id
        logging.info(f'== this_ship_id[{this_ship_id}]')
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue

        all_entities = game_map.nearby_entities_by_distance(ship)
        all_entities_sort_by_distance = entity_list_sort_by_distance(
            all_entities)

        list_empty_planets_sorted_by_distance = get_all_closest_empty_planets_sorted_by_distance(
            all_entities_sort_by_distance)        
        list_enemy_ships_sorted_by_distance = get_all_enemy_ships(
            all_entities_sort_by_distance, allies_ships)

        empty_planets_sorted_by_distance_count = len(list_empty_planets_sorted_by_distance)
        enemy_ships_sorted_by_distance_count = len(list_enemy_ships_sorted_by_distance)
        # There are empty planets -> let dock them
        # Dock planet stage
        if empty_planets_sorted_by_distance_count > 0:
            planet_to_dock = list_empty_planets_sorted_by_distance[0]
            if ship.can_dock(planet_to_dock):
                command_queue.append(ship.dock(planet_to_dock))
            else:
                navigate_command = ship.navigate(ship.closest_point_to(planet_to_dock),
                                                 game_map,
                                                 speed=int(hlt.constants.MAX_SPEED),
                                                 ignore_ships=False)
                if navigate_command:
                    command_queue.append(navigate_command)

        # There is no empty planet -> let kill enemy ships
        # Attack stage
        elif enemy_ships_sorted_by_distance_count > 0:
            target_ship = list_enemy_ships_sorted_by_distance[0]
            navigate_command = ship.navigate(ship.closest_point_to(target_ship),
                                                 game_map,
                                                 speed=int(hlt.constants.MAX_SPEED),
                                                 ignore_ships=True)
            if navigate_command:
                command_queue.append(navigate_command)

    game.send_command_queue(command_queue)
    # TURN END

# GAME END
