from abc import ABC, abstractmethod


class BaseRecommender(ABC):

    @abstractmethod
    def fit(self, urm, tg_playlist, tg_tracks, dataset):
        """
        urm must be an (m, n) csr matrix
        tg_playlist and tg_tracks must be a list of ids
        dataset is the Dataset object
        """
        pass

    @abstractmethod
    def getR_hat(self):
        """
        returns the R_hat as (len(tg_playlist), len(tg_tracks)) csr matrix
        """
        pass
