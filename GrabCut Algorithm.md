# GrabCut Algorithm

## About GrabCut

GrabCut is an interactive foreground extraction algorithm with minimal user interaction. It is used to accurately segment the foreground of an image from the background.

## Algorithm

1. GrabCut works by attributing a label to each pixel in the image (or image segmentation). Each pixel is assigned an alpha value based on whether it is in the foreground region (referred to as $T_F$ where $\alpha = 1$) or the background region (referred to as $T_B$ where $\alpha = 0$) of the image. The rest of the pixels are unknown (referred to as $T_U$), having $0 \leq \alpha \leq 1$, and the algorithm iteratively this $\alpha$ value.  
2. The user starts by manually drawing a rectangular box around the region of interest in the image. Once the region is selected, everything outside the box is hard segmented as the background region ($T_B$ assigned $\alpha=0$), while everything in the box is unknown ($T_U$).
3. A gaussian mixture model (GMM) is applied over the image and this model understands the user input and starts creating labels for unknown pixel values. The unknown pixels are labelled either probable foreground or probable background depending on thier relation with the other hard-labelled pixels in terms of color statistics.
4. The problem is modeled as a task to minimize the following Gibbs energy:
$$E[\alpha, k, \theta, z] = U[\alpha, k, \theta, z] + V[\alpha, z]$$  
where,
$$U[\alpha, k, \theta, z] = \sum_{n}{D(\alpha_n, k_n, \theta, z_n)}$$

$$D(\alpha_n, k_n, \theta, z_n) = -\log{(p(z_n|\alpha_n, k_n, \theta)}\pi(\alpha_n, k_n))$$
$p(.)$ is Gaussian probability distribution, and $\pi(·)$ are mixture weighting coefficients

and,

$$V(\alpha, z) = \gamma \sum_{n,m \in C}{[\alpha_n \neq \alpha_m] exp(-\beta ||{z_n - z_m}||^2 )}$$

where $C$ is a set of neighboring vertices and $\beta$ is a constant.  

5. Based on the above pixel distribution, a graph is generated where these pixels are considered as nodes. Apart from this, we also have two other nodes :
    - Source node : It will be connected to all the foreground pixels after mincut.  
    - Sink node : It will be connected to all the background pixelsa after mincut. 

6. The edges connecting the pixels with the source node or the sink node contains the weights (which is basically the probability of whether the pixel is a foreground one or background one). These weights between the pixels are defined by the edge information or pixel similarity. Mathematically they relate to the Gibbs Energy model as:
    - The edges between two pixel nodes are defined as the entropy term, i.e. for two neighbour nodes $n$ and $m$, weight will be,
    $$ \gamma \ { exp(-\beta\  ||{z_n - z_m}||^2 )}$$
    - The edges between source and sink nodes respectively to the pixel nodes would be calculated from the GMM models trained for foreground and backgrounds or based on user interactions.  
    Hence weights from source node to a pixel $i$ would be,  
    $$ \text{if } i \in T_U \text{ then, } wt =   -\log (score_{BG\_GMM} (i)) $$
    $$ \text{if } i \in T_B \text{ then, } wt =   0 $$
    $$ \text{if } i \in T_F \text{ then, } wt =   \text{high\_weight} $$

    Hence weights from sink node to a pixel $i$ would be,  
    $$ \text{if } i \in T_U \text{ then, } wt =   -\log (score_{FG\_GMM} (i)) $$
    $$ \text{if } i \in T_F \text{ then, } wt =   0 $$
    $$ \text{if } i \in T_B \text{ then, } wt =   \text{high\_weight} $$

7. Now using the mincut algorithm, we cut graph into two parts such that source and sink are in separate partitions with weights of edges cut is minimum. The weights of the edges cut will contribute to the final Gibbs Energy of the segmentation. Hence, the image is segmented into two parts.
8. Now, we have an image where the background and foreground are separated, although the separation is not completely accurate. The user can further imporove this segmentation by manually specifying parts of the image as either background, foreground, probable background, or probable foreground with the help of brush strokes.  We run the entire iterative optimisation and energy minimization again after user refinement.
7.  This process is repeated until we get the extracted foreground image with desired accuracy.

### Model Hyperparameters values
1.  The parameter $\beta$ is defined as $\frac{1}{2<\|z_m-z_n\|^2>}$ (where $z_m, z_n$ are the neighbouring pxiels in the graph), and is calculated by using eight-connectivity.
2.  The weight of the edges connecting sure background pixels to Sink and sure foreground pixels to the Source is set as $10^6$ to prevent these edges from getting removed by the mincut algorithm.
3.  To create a connection between all neighbouring pixels (even if they are at the object boundary), we omit the indicator term from the smoothness energy $V(\underline{\alpha}, \mathbf{z})$.
4.  For each segmentation, we run 1 iteration of the GrabCut algorithm with $\gamma=50$ (as our default values).