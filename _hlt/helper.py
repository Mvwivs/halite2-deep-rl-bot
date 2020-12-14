
from hlt.entity import Position, Ship
from .geom import Point, Seg, min_dist, ps_dist
from hlt.constants import *

def nav(ship, targ, gmap, obs, move_table={}, speed=MAX_SPEED, max_deviation=90):
    dist = ship.calculate_distance_between(targ)
    angle = round(ship.calculate_angle_between(targ))
    speed = speed if (dist >= speed) else int(dist)

    if obs == None:
        dships = [s for s in gmap.get_me().all_ships() if not (s.docking_status == Ship.DockingStatus.UNDOCKED)]
        uships = [s for s in gmap.get_me().all_ships() if (s.docking_status == Ship.DockingStatus.UNDOCKED)]
        obs = [e for e in gmap.all_planets() + dships
                if ship.calculate_distance_between(e)-ship.radius-e.radius <= dist]
        obs.extend([e for e in uships if e != ship
                        and ship.calculate_distance_between(e)-ship.radius-e.radius<=MAX_SPEED*2])


    obs = sorted(obs,key=lambda t:ship.calculate_distance_between(t))
    angs = [int(n/2) if n%2==0 else -int(n/2) for n in range(1,max_deviation*2+2)]

    for d_ang in angs:

        move_ang = (angle+d_ang)%360
        d = Point.polar(speed, move_ang)
        move = Seg(Point(ship.x, ship.y),Point(ship.x, ship.y)+d)

        d = Point.polar(dist,move_ang)
        full_move = Seg(Point(ship.x, ship.y),Point(ship.x, ship.y)+d)
        
        contains = not(move.p2.x < 0 or move.p2.x > gmap.width or move.p2.y < 0 or move.p2.y > gmap.height)
        if not contains:
            continue

        for e in obs:
            collide_dist = ship.radius+e.radius+.000001
            if e in move_table and min_dist(move,move_table[e]) <= collide_dist:
                break
            elif not e in move_table:
                if type(e) == Ship and (e.docking_status == Ship.DockingStatus.UNDOCKED):
                    if ps_dist(Point(e.x, e.y),move)<=collide_dist:
                        break
                elif ps_dist(Point(e.x, e.y),full_move) <=collide_dist:
                    break
        else:
            return ship.thrust(speed,move_ang), move

    return None, None
