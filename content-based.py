from loader import *
from scipy.sparse import *
import numpy as np
import os.path


def main():
    icm = build_tracks_matrix('./data/tracks_final.csv')
    print('ICM matrix evaluated...')
    icm_csr = csr_matrix(icm)
    print(icm_csr.todense())
    if os.path.isfile('./data/icm_sim.npz'):
        sim = load_sparse_matrix('./data/icm_sim.npz')
    else:
        norm = np.sqrt(icm_csr.multiply(icm_csr).sum(0))
        print('Norm done...')
        print(norm)
        icm_bar = icm_csr.multiply(csr_matrix(1 / norm))
        print('Matrix normalized evaluated...')
        sim = (icm_bar.transpose()).dot(icm_bar)
        print('Similarity Matrix evaluated...')
        print(sim.shape)
        sim[sim < 0.1] = 0
        save_sparse_matrix('./data/icm_sim.npz',sim)


if __name__ == '__main__':
    main()
