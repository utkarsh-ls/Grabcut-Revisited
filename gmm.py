import numpy as np


class GaussianMixture:

    def init_with_kmeans(self, X):
        """Computes labels of k-means clusters
        (calls self.fit() function)

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        """
        pass

    def calc_score(self, X, ci)->np.ndarray:
        """Predict probabilities of samples belong to component ci

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        ci : int

        Returns
        -------
        score : array, shape (n_samples,)
        """
        pass
        
    def calc_prob(self, X)->np.ndarray:
        """Predict probability (weighted score) of samples belong to the GMM

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        prob : array, shape (n_samples,)
        """
        pass
    
    def calc_component(self, X)->np.ndarray:
        """Predicts which GMM component the samples belong to

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        comp : array, shape (n_samples,)
        """
        pass

    def fit(self, X, labels) ->None:
        """Computes mean and co-variance

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        labels (Labels for each point) : array, shape (n_samples,)

        Returns
        -------
        None
        """
        pass
