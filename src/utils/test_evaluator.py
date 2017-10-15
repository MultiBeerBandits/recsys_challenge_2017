from src.utils.loader import Dataset
from src.utils.evaluator import *
import random

def main():
    ds = Dataset()
    cross_validation(5, ds.train_final)
    for i in range(0,5):
        current_fold, target_tracks, target_playlist = get_fold(ds)
        recs = {}
        for pl in target_playlist:
            recs[pl] = random.sample(target_tracks, 5)
        evaluate_fold(recs)
    print("Average MAP@5 " + str(get_mean_map()))


if __name__ == '__main__':
    main()