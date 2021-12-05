from typing import Tuple
import numpy as np
import igraph as ig
from gmm import GMM
from enum import Enum


class MaskValues:
    bg = 0
    fg = 1
    pr_bg = 2
    pr_fg = 3


# These are the best no of GMM components K
# that were mentioned in paper
gmm_components = 5
gamma = 50


class Graph:
    strong_edges_wt = 1e6
    weak_edges_wt = 0

    def __init__(self, img_shape) -> None:
        self.img_shape = img_shape
        self.n_nodes = img_shape[0]*img_shape[1] + 2
        self.source = self.n_nodes - 2      # FG terminal
        self.sink = self.n_nodes - 1       # BG terminal

    def set_edges(self, img, mask, fg_gmm: GMM, bg_gmm: GMM):
        r, c = mask.shape
        # nodes (only pixels not source and sink)
        px_nodes = np.arange(self.n_nodes-2).reshape(mask.shape)
        edges = []
        weights = []
        exp_length = 0

        def len_check():
            assert len(edges) == len(weights), ""\
                f"e_len: {len(edges)}, w_len: {len(weights)}"
            assert len(edges) == exp_length, ""\
                f"e_len: {len(edges)}, exp_len: {exp_length}"

        # color diff for horizontal edges
        # -
        edges.extend(np.hstack((px_nodes[:, :-1].reshape(-1, 1), px_nodes[:, 1:].reshape(-1, 1))))
        col_del_hori = np.linalg.norm(img[:, :-1] - img[:, 1:], axis=-1).flatten()**2
        # |
        edges.extend(np.hstack((px_nodes[:-1, :].reshape(-1, 1), px_nodes[1:, :].reshape(-1, 1))))
        col_del_vert = np.linalg.norm(img[:-1, :] - img[1:, :], axis=-1).flatten()**2
        # \
        edges.extend(np.hstack((px_nodes[:-1, :-1].reshape(-1, 1), px_nodes[1:, 1:].reshape(-1, 1))))
        col_del_dig1 = np.linalg.norm(img[:-1, :-1] - img[1:, 1:], axis=-1).flatten()**2
        # /
        edges.extend(np.hstack((px_nodes[:-1, 1:].reshape(-1, 1), px_nodes[1:, :-1].reshape(-1, 1))))
        col_del_dig2 = np.linalg.norm(img[:-1, 1:] - img[1:, :-1], axis=-1).flatten()**2

        del_mean = np.hstack(
            (col_del_hori, col_del_vert, col_del_dig1, col_del_dig2)).mean()
        beta = 1/(2 * del_mean)

        def smoothness(col_diff):
            return gamma * np.exp(-beta * col_diff)

        weights.extend(smoothness(col_del_hori))
        exp_length += r*(c-1)
        weights.extend(smoothness(col_del_vert))
        exp_length += c*(r-1)
        weights.extend(smoothness(col_del_dig1))
        exp_length += (r-1)*(c-1)
        weights.extend(smoothness(col_del_dig2))
        exp_length += (r-1)*(c-1)

        len_check()

        # already flattened array
        sure_fg_nodes = px_nodes[mask == MaskValues.fg]
        sure_bg_nodes = px_nodes[mask == MaskValues.bg]
        pr_nodes = px_nodes[(mask == MaskValues.pr_bg) |
                            (mask == MaskValues.pr_fg)]
        pr_nodes_cols = img[(mask == MaskValues.pr_bg) |
                            (mask == MaskValues.pr_fg)]

        def single_to_many_update(source, dests, wt):
            edges.extend(
                np.hstack((np.full(dests.size, source).reshape(-1, 1), dests.reshape(-1, 1))))
            weights.extend(np.ones_like(dests)*wt)

        # sure fg - source
        single_to_many_update(self.source, sure_fg_nodes, self.strong_edges_wt)

        # sure bg - source
        single_to_many_update(self.source, sure_bg_nodes, self.weak_edges_wt)

        # pr - source
        bg_scores = bg_gmm.prob(pr_nodes_cols)
        single_to_many_update(self.source, pr_nodes, -np.log(bg_scores))

        exp_length += r*c
        len_check()

        # sure fg - sink
        single_to_many_update(self.sink, sure_fg_nodes, self.weak_edges_wt)

        # sure bg - sink
        single_to_many_update(self.sink, sure_bg_nodes, self.strong_edges_wt)

        # pr - sink
        fg_scores = fg_gmm.prob(pr_nodes_cols)
        single_to_many_update(self.sink, pr_nodes, -np.log(fg_scores))

        exp_length += r*c
        len_check()

        assert exp_length == (6*r*c - 3*(r + c) + 2)

        self.edges = edges
        self.weights = weights

    def mincut(self):
        graph = ig.Graph(self.n_nodes)
        graph.add_edges(self.edges)
        ret = graph.st_mincut(self.source, self.sink, self.weights)
        if self.source in ret.partition[0]:
            fg_nodes, bg_nodes = ret.partition[0], ret.partition[1]
        else:
            fg_nodes, bg_nodes = ret.partition[1], ret.partition[0]

        fg_nodes.remove(self.source)
        bg_nodes.remove(self.sink)
        return fg_nodes, bg_nodes


def GrabCut(img, init_mask=None, rect=None, n_itrs=1):
    img = np.array(img, dtype=np.int)

    if rect:
        mask = np.full(img.shape[:2], MaskValues.bg)
        mask[rect[1]:rect[1] + rect[3], 
            rect[0]:rect[0] + rect[2]] = MaskValues.pr_fg
    else:
        assert init_mask is not None
        mask = np.array(init_mask, np.int)

    pr_mask = ((mask == MaskValues.pr_bg) | (mask == MaskValues.pr_fg))
    pr_ids = np.arange(mask.size).reshape(mask.shape)[pr_mask]
    graph = Graph(mask.shape)

    def fit_gmms():
        fg_mask = (mask == MaskValues.fg) | (mask == MaskValues.pr_fg)
        bg_mask = (mask == MaskValues.bg) | (mask == MaskValues.pr_bg)
        # fit is called inside __init__
        return GMM(img[fg_mask]), GMM(img[bg_mask])

    for _ in range(n_itrs):
        fg_gmm, bg_gmm = fit_gmms()
        graph.set_edges(img, mask, fg_gmm, bg_gmm)
        fg_ids, bg_ids = graph.mincut()
        mask[pr_mask] = np.where(
            np.isin(pr_ids, fg_ids), MaskValues.pr_fg, MaskValues.pr_bg)

    return mask
