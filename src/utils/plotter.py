from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from src.utils.cluster import build_user_cluster


def plot_km_sse():
    durations = []
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            duration = row['duration']
            if duration is not None and duration != '' and float(duration) != -1:
                durations.append(float(duration))
    durations = np.array(durations)
    sse = []
    for i in range(8, 30):
        k_m = KMeans(n_clusters=i)
        dur_cluster = k_m.fit_predict(np.reshape(durations, (-1, 1)))
        print((k_m.inertia_ / len(durations)) / 1e7)
        sse.append((k_m.inertia_ / len(durations)) / 1e7)
    fig, ax = plt.subplots()
    plt.plot(range(8, 30), sse, 'ro')
    plt.show()


def km_sse_playcount():
    playcounts = []
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            playcount = row['playcount']
            if playcount is not None and playcount != '' and float(playcount) != -1:
                playcounts.append(float(playcount))
    playcounts = np.array(playcounts)
    sse = []
    for i in range(10, 40):
        k_m = KMeans(n_clusters=i)
        play_cluster = k_m.fit_predict(np.reshape(playcounts, (-1, 1)))
        print((k_m.inertia_ / len(playcounts)) / 1e4)
        sse.append((k_m.inertia_ / len(playcounts)) / 1e4)
    fig, ax = plt.subplots()
    plt.plot(range(10, 40), sse, 'ro')
    plt.ylabel("SSE for playcount")
    plt.xlabel("Cluster Size")
    plt.title("Playcount SSE")
    plt.show()


def km_sse_created_at():
    created_ats = []
    with open('./data/playlists_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            created_at = row['created_at']
            created_ats.append(float(created_at))
    created_ats = np.array(created_ats)
    sse = []
    for i in range(10, 40):
        k_m = KMeans(n_clusters=i)
        created_cluster = k_m.fit_predict(np.reshape(created_ats, (-1, 1)))
        print((k_m.inertia_ / len(created_ats)) / 1e4)
        sse.append((k_m.inertia_ / len(created_ats)) / 1e4)
    fig, ax = plt.subplots()
    plt.plot(range(10, 40), sse, 'ro')
    plt.ylabel("SSE for Created At")
    plt.xlabel("Cluster Size")
    plt.title("Create at SSE")
    plt.show()


def km_sse_num_rating():
    """
    25 cluster seems good
    """
    dataset = Dataset()
    urm = dataset.build_train_matrix()
    num_rating = np.ravel(urm.sum(axis=0))
    sse = []
    for i in range(10, 40):
        k_m = KMeans(n_clusters=i)
        rating_cluster = k_m.fit_predict(np.reshape(num_rating, (-1, 1)))
        print((k_m.inertia_ / len(num_rating)) / 1e4)
        sse.append((k_m.inertia_ / len(num_rating)) / 1e4)
    fig, ax = plt.subplots()
    plt.plot(range(10, 40), sse, 'ro')
    plt.ylabel("SSE for Created At")
    plt.xlabel("Cluster Size")
    plt.title("Create at SSE")
    plt.show()


def km_sse_user_num_rating():
    """
    25 cluster seems good
    """
    dataset = Dataset()
    urm = dataset.build_train_matrix()
    num_rating = np.ravel(urm.sum(axis=1))
    sse = []
    for i in range(10, 40):
        k_m = KMeans(n_clusters=i)
        rating_cluster = k_m.fit_predict(np.reshape(num_rating, (-1, 1)))
        print((k_m.inertia_ / len(num_rating)) / 1e4)
        sse.append((k_m.inertia_ / len(num_rating)) / 1e4)
    fig, ax = plt.subplots()
    plt.plot(range(10, 40), sse, 'ro')
    plt.ylabel("SSE for Created At")
    plt.xlabel("Cluster Size")
    plt.title("Create at SSE")
    plt.show()


def plot_duration():
    durations = []
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            duration = row['duration']
            if duration is not None and duration != '' and float(duration) != -1:
                durations.append(float(duration))
    durations = np.array(durations)
    fig, ax = plt.subplots()
    ax.scatter(durations, np.ones_like(durations))
    plt.show()
    # Do K-means
    dur_cluster = KMeans(n_clusters=22).fit_predict(
        np.reshape(durations, (-1, 1)))
    fig, ax = plt.subplots()
    ax.scatter(durations[:100], np.zeros_like(
        durations[:100]), c=dur_cluster[:100])
    plt.show()


def plot_playcount():
    playcounts = []
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            playcount = row['playcount']
            if playcount is not None and playcount != '' and float(playcount) != -1:
                playcounts.append(float(playcount))
    playcounts = np.array(playcounts)
    print(len([x for x in playcounts if x > 2e8]))
    fig, ax = plt.subplots()
    ax.scatter(playcounts, np.ones_like(playcounts))
    plt.show()
    # Do K-means
    play_cluster = KMeans(n_clusters=20).fit_predict(
        np.reshape(playcounts, (-1, 1)))
    fig, ax = plt.subplots()
    ax.scatter(playcounts[:100], np.zeros_like(
        playcounts[:100]), c=play_cluster[:100])
    plt.show()


def plot_tags():
    # load tags
    tag_dict = {}
    tag_index = 0
    track_index = 0
    # first reading of tags
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            tags = parse_csv_array(row['tags'])
            for tag in tags:
                if tag not in tag_dict:
                    tag_dict[tag] = tag_index
                    tag_index += 1
            track_index += 1
    tag_M = lil_matrix((track_index + 1, tag_index + 1))
    track_index = 0
    with open('./data/tracks_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            tags = parse_csv_array(row['tags'])
            for tag in tags:
                tag_col = tag_dict[tag]
                tag_M[track_index, tag_col] = 1
            track_index += 1
    tag_svd = TruncatedSVD(n_components=3).fit_transform(tag_M)
    tag_cluster = KMeans(n_clusters=10).fit_predict(tag_svd)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tag_svd[:1000 , 0], tag_svd[:1000 , 1], tag_svd[:1000 , 2], c=tag_cluster[:1000])
    plt.show()


def plot_created_at():
    created_ats = []
    with open('./data/playlists_final.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            # duration
            created_at = row['created_at']
            created_ats.append(float(created_at))
    created_ats = np.array(created_ats)
    fig, ax = plt.subplots()
    ax.scatter(created_ats, np.ones_like(created_ats))
    plt.show()
    # Do K-means
    created_at_cluster = KMeans(n_clusters=20).fit_predict(
        np.reshape(created_ats, (-1, 1)))
    fig, ax = plt.subplots()
    ax.scatter(created_ats[:100], np.zeros_like(
        created_ats[:100]), c=created_at_cluster[:100])
    plt.show()


def plot_km_user_cluster():
    ds = Dataset()
    urm = ds.build_train_matrix()
    ucm = ds.build_ucm()
    icm = ds.build_icm()

    sse = []
    for i in range(10, 40):
        X = getUCM(urm, icm, ucm)
        k_m = KMeans(n_clusters=i)
        cluster = k_m.fit_predict(X)
        print((k_m.inertia_ / len(cluster)))
        sse.append((k_m.inertia_ / len(cluster)))
    fig, ax = plt.subplots()
    plt.plot(range(10, 40), sse, 'ro')
    plt.ylabel("SSE for Created At")
    plt.xlabel("Cluster Size")
    plt.title("Create at SSE")
    plt.show()


def getUCM(urm, icm, ucm):
    # first build the user feature matrix: UxF (features of rated items)
    ufm = urm.dot(icm.transpose())

    # normalize dividing by rating of each user
    Nu = urm.sum(axis=1)
    Iu = np.copy(Nu)
    # save from divide by zero!
    Iu[Iu == 0] = 1
    # since we have to divide the upm get the reciprocal of this vector
    Iu = np.reciprocal(Iu)
    # multiply the ufm by Iu. Normalize UFM
    ufm = ufm.multiply(Iu)

    # add a column for the number of rating
    ufm = hstack([ufm, Nu], format='csr')

    # stack all matrices horizontally
    ucm = hstack([urm, ufm, ucm.transpose()], format='csr')

    return ucm


def plot_num_rating():
    dataset = Dataset()
    urm = dataset.build_train_matrix()
    num_rating = np.ravel(urm.sum(axis=0))
    print(num_rating.shape)
    fig, ax = plt.subplots()
    ax.scatter(num_rating, np.ones_like(num_rating))
    plt.show()
    # Do K-means
    rating_cluster = KMeans(n_clusters=22).fit_predict(
        np.reshape(num_rating, (-1, 1)))
    fig, ax = plt.subplots()
    ax.scatter(num_rating[:100], np.zeros_like(
        num_rating[:100]), c=rating_cluster[:100])
    plt.show()


def visualize_2d(x, y, x_label, y_label, title):
    fig, ax = plt.subplots()
    plt.axis([0, len(x), 0, 0.3])
    plt.grid(True)
    ax.scatter(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(title)



if __name__ == '__main__':
    km_sse_user_num_rating()
