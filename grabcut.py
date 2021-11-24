from typing import Tuple
import numpy as np
import igraph as ig
from gmm import GMM
import types

MASK_VALUE = {

    "bg": 0,
    "fg": 1,
    "pr_bg": 2,
    "pr_fg": 3,
}

# These are the best no of GMM components K
# that were mentioned in paper
gmm_components = 5
gamma = 50
beta = 0



def GrabCut(img, init_mask, rect=None):
    # If we don’t define dtype in the syntax, it is defined by default inferred from the input data.
    img = np.asarray(img, dtype=np.float64)
    rows = img.shape[0]
    cols = img.shape[1]
    # _=img.shape[2]
    mask = init_mask
    if rect is None:
        pass
    else:
        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = MASK_VALUE['pr_fg']
    bgd_indexes, fgd_indexes = classify_pixel(mask)

    print("bg count = ", bgd_indexes[0].size, end="")
    print("fg count = ", fgd_indexes[0].size)
    bgd_gmm, fgd_gmm = None, None
    smootness = {}
    smootness["left_V"] = make_empty_array(rows, cols-1)
    smootness["up_V"] = make_empty_array(rows-1, cols)
    smootness["upleft_V"] = make_empty_array(rows-1, cols-1)
    smootness["upright_V"] = make_empty_array(rows-1, cols-1)
    comp_indxs = make_empty_array(rows, cols).astype('uint32')
    gc={}
    gc["graph"]=None
    gc["graph_capacity"] = None  # Edge capacities
    gc["source"] = rows*cols    # "object" terminal S
    gc["sink"] = gc["source"]+1    # "background" terminal T
    smoothness = calc_beta_smoothness(img, rows, cols, smootness)

    # intitializing gmm
    bgd_gmm = GMM(img[bgd_indexes])
    fgd_gmm = GMM(img[fgd_indexes])

    #RUN wala karna hai ab ....
    # RUN part
    #for number of grabcut iterations
    ITERATION_CNT=1
    for _ in range(ITERATION_CNT):
        # Assigning GMM Components -> Ist step
        # This is the first part of the 3 steps of paper
        # It basically assigns GMM components to pixels 
        comp_indxs[bgd_indexes]=bgd_gmm.component(img[bgd_indexes])
        comp_indxs[fgd_indexes]=fgd_gmm.component(img[fgd_indexes])


        # This function performs the second step of image 3 in paper -> Step 2
        # This learn GMM Parameters from data z 
        bgd_gmm.fit(img[bgd_indexes],comp_indxs[bgd_indexes])
        fgd_gmm.fit(img[fgd_indexes],comp_indxs[fgd_indexes])
        updated_var=construct_gc_graph(mask,gc,bgd_gmm,fgd_gmm,rows,cols,smoothness,img)
        mask=updated_var["mask"]
        gc=updated_var["gc"]
        bgd_gmm=updated_var["bgd_gmm"]
        fgd_gmm=updated_var["fgd_gmm"]
        rows=updated_var["rows"]
        cols=updated_var["cols"]
        smoothness=updated_var["smoothness"]
        img=updated_var["img"]



    

def classify_pixel(mask):
    bgd_indexes = np.where(np.logical_or(
        mask == MASK_VALUE['bg'], mask == MASK_VALUE['pr_bg']))
    fgd_indexes = np.where(np.logical_or(
        mask == MASK_VALUE['fg'], mask == MASK_VALUE['pr_fg']))

    if bgd_indexes[0].size <= 0:
        print("Incorrect value : error in bg indexes")

    if fgd_indexes[0].size <= 0:
        print("Incorrect value : error in fg indexes")

    return bgd_indexes, fgd_indexes

def make_empty_array(r, c):
    return np.empty((r, c))


def calc_beta_smoothness(img, rows, cols, smoothness):
    # @VEDANSH PLEASE CORRECT THESE 4 LINES
    _left_diff = img[:, 1:] - img[:, :-1]
    _up_diff = img[1:, :] - img[:-1, :]
    _upleft_diff = img[1:, 1:] - img[:-1, :-1]
    _upright_diff = img[1:, :-1] - img[:-1, 1:]

    sum1 = np.sum(np.square(_left_diff))
    sum2 = np.sum(np.square(_up_diff))
    sum3 = np.sum(np.square(_upleft_diff))
    sum4 = np.sum(np.square(_upright_diff))
    Σ = sum1+sum2+sum3+sum4
    global beta
    global gamma
    beta = 1 / (2 * Σ / (4 * cols * rows - 3 * (cols + rows) + 2))
    print(" Beta : ", beta)
    # Each pixel has 4 neighbors (left, upleft, up, upright)
    # The 1st column doesn't have left, upleft and the last column doesn't have upright
    # The first row doesn't have upleft, up and upright
    # The first and last pixels in the 1st row are removed twice

    #now we have used fomula 11 of the research paper for V (smoothness)
    smoothness['left_V'] = gamma * np.exp(-beta * np.sum( _left_diff**2, axis=2))
    smoothness['upleft_V'] = gamma * \
        np.exp(-beta * np.sum(_upleft_diff** 2, axis=2)) * (1/np.sqrt(2))
    smoothness['up_V'] = gamma * np.exp(-beta * np.sum(_up_diff** 2, axis=2))
    smoothness['upright_V'] = gamma * \
        np.exp(-beta * np.sum(_upright_diff** 2, axis=2)) * (1/np.sqrt(2))
    return smoothness


def construct_gc_graph(mask,gc,bgd_gmm:GMM,fgd_gmm:GMM,rows,cols,smoothness,img):
    edges,gc["graph_capacity"]=[],[]
    mask=mask.reshape(-1) #Edited in mask : NOTE FOR ERROR
    fg_idx=np.where(mask==MASK_VALUE["fg"])
    bg_idx=np.where(mask==MASK_VALUE["bg"])
    pr_idx=np.where(np.logical_or(mask==MASK_VALUE["pr_fg"],mask==MASK_VALUE["pr_bg"]))

    print("bg count = ", len(bg_idx[0]), end="")
    print("fg count = ", len(fg_idx[0]),end='')
    print("uncertanity count = ", len(pr_idx[0]))
    global gamma

    #T-LINKS

    edges.extend(
        list(zip([gc["source"]] * pr_idx[0].size, pr_idx[0])))
    _D = -np.log(bgd_gmm.prob(img.reshape(-1, 3)[pr_idx]))
    gc["graph_capacity"].extend(_D.tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")


    edges.extend(
        list(zip([gc["sink"]] * pr_idx[0].size, pr_idx[0])))
    _D = -np.log(fgd_gmm.prob(img.reshape(-1, 3)[pr_idx]))
    gc["graph_capacity"].extend(_D.tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")


    edges.extend(
        list(zip([gc["source"]] * bg_idx[0].size, bg_idx[0])))
    gc["graph_capacity"].extend([0] * bg_idx[0].size)
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")

    edges.extend(
        list(zip([gc["sink"]] * bg_idx[0].size, bg_idx[0])))
    gc["graph_capacity"].extend([9 * gamma] * bg_idx[0].size)
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")

    edges.extend(
        list(zip([gc["source"]] * fg_idx[0].size, fg_idx[0])))
    gc["graph_capacity"].extend([9 * gamma] * fg_idx[0].size)
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")

    edges.extend(
        list(zip([gc["sink"]] * fg_idx[0].size, fg_idx[0])))
    gc["graph_capacity"].extend([0] * fg_idx[0].size)
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in adding edges ")
    # print(len(edges))
    # N-LINKS
    img_index=np.arange(rows*cols).astype('uint32')
    img_index=img_index.reshape(rows,cols)

    mask1 = img_index[:, 1:].reshape(-1)
    mask2 = img_index[:, :-1].reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc["graph_capacity"].extend(smoothness["left_V"].reshape(-1).tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in N-Links ")

    mask1 = img_index[1:, 1:].reshape(-1)
    mask2 = img_index[:-1, :-1].reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc["graph_capacity"].extend(smoothness["upleft_V"].reshape(-1).tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in N-Links ")

    mask1 = img_index[1:, :].reshape(-1)
    mask2 = img_index[:-1, :].reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc["graph_capacity"].extend(smoothness["up_V"].reshape(-1).tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in N-Links ")

    mask1 = img_index[1:, :-1].reshape(-1)
    mask2 = img_index[:-1, 1:].reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc["graph_capacity"].extend(smoothness["upright_V"].reshape(-1).tolist())
    if len(edges)!=len(gc["graph_capacity"]):
        print("Error in N-Links ")

    cap=4*(rows*cols)-3*(rows+cols)+2*(rows*cols)+2
    assert len(edges) == cap

    gc["graph"] = ig.Graph(rows*cols+2)
    gc["graph"].add_edges(edges)

    # ESTIMATING SEGMENTATION -> STEP 3
    # This is basically the 3rd part of figure 3

    #we are calculating the mincut 
    
    mincut=gc["graph"].st_mincut(gc["source"],gc["sink"],gc["graph_capacity"])
    print("bg pixels = ", len(mincut.partition[0]), end="")
    print("fg pixels = ", len(mincut.partition[1]))
    mask=mask.reshape(rows,cols)
    pr_idx = (mask == MASK_VALUE['pr_fg']) | (mask == MASK_VALUE['pr_bg'])
    print("shapes ",pr_idx.shape,mask.shape)
    img_index = np.arange(rows * cols).astype("uint32")
    img_index=img_index.reshape(rows, cols)
    print(img_index[pr_idx].shape)
    print("mask pr " ,mask[pr_idx].shape)
    print(np.isin(img_index[pr_idx], mincut.partition[0]).shape)
    # mask[np.isin(img_index[pr_idx], mincut.partition[0])]=MASK_VALUE['pr_fg']
    # mask[np.isin(img_index[pr_idx], mincut.partition[1])]=MASK_VALUE['pr_bg']
    mask[pr_idx] = np.where(np.isin(img_index[pr_idx], mincut.partition[0]),MASK_VALUE['pr_fg'], MASK_VALUE['pr_bg'])
    
    classify_pixel(mask)

    updated_variables={
        "mask":mask,
        "gc":gc,
        "bgd_gmm":bgd_gmm,
        "fgd_gmm":fgd_gmm,
        "rows":rows,
        "cols":cols,
        "smoothness":smoothness,
        "img":img
    }
    return updated_variables
