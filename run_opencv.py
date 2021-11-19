from window_manager import WindowManager
from cv2 import grabCut

if __name__ == '__main__':
    wm = WindowManager("messi5.jpg", grabCut)
    wm.run()

