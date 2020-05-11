import cv2
import threading

def start():
    img = cv2.imread('E:\\test.jpg')
    print(img.shape)

threading.Thread(target=start).start()

