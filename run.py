from window_manager import WindowManager
from grabcut import GrabCut

def grabcut_fn(img,mask,rect,tmp1,tmp2,itr,mode):
    if mode==1:
        rect=None
    
    mask[:] = GrabCut(img,mask,rect)
    # gc.run(skip_learn_GMMs=(mode==0))
    # mask[:]=gc.mask
    # print("ingcu",mask.sum(),mask.std())



import sys
if __name__ == '__main__':
    wm = WindowManager(sys.argv[1], grabcut_fn)
    wm.run()

