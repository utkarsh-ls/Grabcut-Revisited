from typing import Tuple
import numpy as np
import igraph as ig

class GrabCut:
    def __init__(self, img, mask, rect=None, gmm_components=5):
        '''Initializes the GMMs and calculate beta smoothness
        Paremeters:
        - img: np.ndarray (rows,cols,3), values in [0,255]
        - mask: np.ndarray (rows, cols), values in {0,1,2,3}
                0: gauranteed background
                1: gauranteed foreground
                2: probably background
                3: probably foreground
        - rect: 4-tuple needed when initializing first time
        - gmm_components: no of components GMM considers
        '''
        pass

    def _determine_edges_weights(self) -> None:
        ''' 
            Intialize the weights of the edges for mincut.
            Uses beta smoothing concept.
        '''
        pass
    
    @property
    def classification(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Returns two tuple: indices of foreground pixels and indices of background pixels
        '''
        pass

    def _assign_gmm_components(self) -> None:
        '''  
            Assigns GMM components to pixels for both models
        '''
        pass
    def _learn_gmm(self) -> None:
        '''  
            Learns GMM parameters for both models
        '''
        pass

    def _build_graph(self) -> None:
        '''  Builds graph data structure
        '''

        # self.graph = ig.Graph( no_of_nodes )
        # self.graph.add_edges(unweighted_edges_list)
        pass

    def _apply_segmentation(self) -> None:
        '''Segments graph using mincut and updates mask accordingly
        '''

        # self.graph.st_mincut()

        pass

    @property
    def energy(self)->float:
        '''
            Calculates energy value.
        '''
        pass

    def run(self, iters:int, learn_gmm:bool=True)->None:
        '''
            Main function to run algorithm.
            Will update self.mask appropiately
        '''
        pass
