
from pathlib import Path
import gc
import numpy as np

def combine(files, result):
    # 
    

    y = []

    for i, f in enumerate(files):
        y.append(np.load(f))
        print(f'{i=}, filesize: {y[-1].shape[0]}, {f=}')

    total = 0
    for batch in y:
        total += batch.shape[0]

    print(f'{total=}')
    gc.collect()

    res = np.empty((total, y[0].shape[1]))

    end = 0
    for batch in y:
        res[end:end + batch.shape[0],] = batch
        end += batch.shape[0]

    np.save(result, res, allow_pickle=False)

t = 'test'
train_folder = f'/home/vova/Downloads/hlt_client/hlt_client/{t}_half/'
output_files = list(Path(train_folder).glob('*_outputs.npy'))
feature_files = list(Path(train_folder).glob('*_features.npy'))
output_files.sort()
feature_files.sort()
combine(feature_files, f'/home/vova/Downloads/hlt_client/hlt_client/data/{t}_comb_features_half.npy')
combine(output_files, f'/home/vova/Downloads/hlt_client/hlt_client/data/{t}_comb_outputs_half.npy')
