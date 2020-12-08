
import sys
from pathlib import Path

import numpy as np

def load_at_least(filename_list, batch_size):
    data = None
    for i, f in enumerate(filename_list):
        if data is None:
            data = np.load(f)
        else:
            data = np.concatenate((data, np.load(f)))
        if data.shape[0] >= batch_size:
            return data, i + 1
    return data, None
        

def data_generator(folder, batch_size):
    while True:
        features_files = list(Path(folder).glob('*_features.npy'))
        outputs_files = list(Path(folder).glob('*_outputs.npy'))
        features_files.sort()
        outputs_files.sort()

        i = 1
        features = np.load(features_files[0])
        outputs = np.load(outputs_files[0])

        x = np.zeros((batch_size, features.shape[1]))
        y = np.zeros((batch_size, outputs.shape[1]))

        end = 0
        while True:
            remaining = features.shape[0] - end
            if remaining < batch_size:
                x[0:remaining] = features[end:]
                y[0:remaining] = outputs[end:]
                # load additional data
                features, cnt = load_at_least(features_files[i:], batch_size - remaining)
                outputs, _ = load_at_least(outputs_files[i:], batch_size - remaining)
                if features is None:
                    yield x, y
                    break

                if cnt is None:
                    x[remaining:batch_size] = features[0:]
                    y[remaining:batch_size] = outputs[0:]
                    yield x, y
                    break
                
                i += cnt
                x[remaining:batch_size] = features[0:batch_size - remaining]
                y[remaining:batch_size] = outputs[0:batch_size - remaining]
                end = batch_size - remaining
                yield x, y

            else:
                x = features[end:end + batch_size]
                y = outputs[end:end + batch_size]
                end += batch_size
                yield x, y

def data_generator_memory(folder, batch_size):
    features_files = list(Path(folder).glob('*_features.npy'))
    outputs_files = list(Path(folder).glob('*_outputs.npy'))

    features_shape = np.load(features_files[0]).shape
    outputs_shape = np.load(outputs_files[0]).shape

    x = np.zeros((1, features_shape[1]))
    y = np.zeros((1, outputs_shape[1]))

    for f in features_files:
        x = np.concatenate((x, np.load(f)))
    for f in outputs_files:
        y = np.concatenate((y, np.load(f)))

    while True:
        end = 0
        while end < x.shape[0]:
            f = x[end:end + batch_size]
            o = y[end:end + batch_size]
            end += batch_size
            yield f, o


if __name__ == "__main__":
    for x,y in data_generator_memory('/home/vova/Downloads/hlt_client/hlt_client/tmp/', 32):
        print(x, y)
        # pass
