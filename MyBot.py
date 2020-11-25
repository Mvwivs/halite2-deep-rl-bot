
import time
import hlt
import halite_env

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}")

for i in range(0, 100):
    reward = 0
    game_map = env.reset()
    while True:
        command_queue = []
        for ship in game_map.get_me().all_ships():
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                continue

            for planet in game_map.all_planets():
                if planet.is_owned():
                    continue

                if ship.can_dock(planet):
                    command_queue.append(ship.dock(planet))
                else:
                    navigate_command = ship.navigate(
                        ship.closest_point_to(planet),
                        game_map,
                        speed=int(hlt.constants.MAX_SPEED/2),
                        ignore_ships=True)
                    if navigate_command:
                        command_queue.append(navigate_command)
                break

        game_map, reward, done, _ = env.step(command_queue)
        if done:
            break

env.close()
print("done")