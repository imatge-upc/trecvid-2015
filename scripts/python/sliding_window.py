import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from scipy.misc import lena
import cv2
import Image
from get_params import get_params
from matplotlib import pyplot as plt

def slide(boxes,height,width,stpx,stpy,sx,sy):

    for i in range(0,height,stpy):
        for j in range(0,width,stpx):

            boxes.append([i, j, min(i+sy,height), min(j+sx,width)])
    return boxes
    
def get_boxes(params):
    
    height = params['height']
    width = params['width']
    
    sizes_x = [32,64,128,256,512]
    sizes_y = [32,64,128,256,512]
    boxes = []

    for sx in sizes_x:
        for sy in sizes_y:

            stpx = sx/2
            stpy = sy/2
            boxes = slide(boxes,height,width,stpx,stpy,sx,sy)
    return boxes

if __name__ == "__main__":

    params = get_params()
    boxes = get_boxes(params)
    
    print np.shape(boxes)

