
import glob
import os

replay_dir = "/home/vova/Downloads/hlt_client/hlt_client/replays (copy)"

replay_list = glob.glob(replay_dir + "/*")
for i, replay in enumerate(replay_list):
    player_cnt = 0
    with open(replay, "r") as f:
        data = f.read().replace('\n', '')
        name_start = data.find('\"player_names\"')
        name_end = data.find(']',name_start)
        player_cnt = data[name_start:name_end].count(',') + 1
    print(f'{i} of {len(replay_list)} {player_cnt=} for {replay}')
    if player_cnt != 2:
        os.remove(replay)
