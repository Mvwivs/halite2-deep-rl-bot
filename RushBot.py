
import hlt
import logging

from _hlt import helper

game = hlt.Game("Rush")
logging.info("Starting my Rush bot!")

def nearest(ship, enemies, filter):
    min_dist = 1000
    min_id = None
    for i, enemy in enumerate(enemy_ships):
        if not filter(i, enemy):
            continue
        dist = ship.calculate_distance_between(enemy)
        if dist < min_dist:
            min_dist = dist
            min_id = i
    return min_id, min_dist

while True:
    game_map: hlt.game_map.Map = game.update_map()

    command_queue = []
    player_id = game_map.get_me().id    

    my_ships = [ship for ship in game_map.get_me().all_ships()]
    enemy_ships = [ship for player in game_map.all_players() for ship in player.all_ships() if player.id != player_id]

    if len(enemy_ships) == 0:
        game.send_command_queue(command_queue)

    move_table = {}
    assigned = []
    for ship in my_ships:
        navigate_command = None

        min_id, min_dist = nearest(ship, enemy_ships, 
            lambda i, e: (i not in assigned and (e.docking_status != e.DockingStatus.UNDOCKED)))
        if min_id is not None:
            navigate_command, move = helper.nav(ship, ship.closest_point_to(enemy_ships[min_id], min_distance=4), game_map, None, move_table)
            if move:
                move_table[ship] = move
                assigned.append(min_id)

        else:
            min_id, min_dist = nearest(ship, enemy_ships, 
                lambda i, e: (e.docking_status != e.DockingStatus.UNDOCKED))
            
            if min_id is not None:
                navigate_command, move = helper.nav(ship, ship.closest_point_to(enemy_ships[min_id], min_distance=4), game_map, None, move_table)
                if move:
                    move_table[ship] = move

            else:
                min_id, min_dist = nearest(ship, enemy_ships, 
                    lambda i, e: True)
                
                if min_id is not None:
                    navigate_command, move = helper.nav(ship, ship.closest_point_to(enemy_ships[min_id], min_distance=4), game_map, None, move_table)
                    if move:
                        move_table[ship] = move

        if navigate_command is not None:
            command_queue.append(navigate_command)

    game.send_command_queue(command_queue)
