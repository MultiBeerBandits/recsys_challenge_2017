from loader import *


def main():
    dataset = Dataset()
    print(dataset.playlists_final,
          dataset.target_playlists,
          dataset.tracks_final,
          dataset.train_final,
          dataset.target_tracks, sep='\n')


if __name__ == '__main__':
    main()
