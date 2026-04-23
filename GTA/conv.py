import numpy as np, pickle

for f in ['MSL_train', 'MSL_test', 'MSL_test_label']:
    data = np.load(f'./data/MSL/{f}.npy')
    with open(f'./data/MSL/{f}.pkl', 'wb') as out:
        pickle.dump(data, out)