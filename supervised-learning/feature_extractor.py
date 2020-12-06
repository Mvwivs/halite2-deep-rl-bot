
import numpy as np
import json

def parse_replay(path):
    with open(path, "r") as f:
        replay = json.loads(f.readline()[2:-1])
        print(f'{replay=}')

parse_replay("./test_replay/halite-2-gold-replays_replay-20180129-000000%2B0000--3975407872-240-160-1517183992")
