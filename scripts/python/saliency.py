from get_params import get_params
import sys
import os
import numpy as np
import matplotlib.pylab as plt
import pickle
from select_samples import find_coordinates
import cv2

''' Obtain Saliency map for query images using SalNet '''

params = get_params() # check get_params.py in the same directory to see the parameters

sys.path.insert(0,os.path.join(params['caffe_path'],'python'))
import caffe

def init_net(params):
    
    deploy_file = os.path.join(params['saliency_model'],'deploy.prototxt')
    model_file = os.path.join(params['saliency_model'],'model.caffemodel')
    
    # I am using the mean file from caffenet...but I guess we could use a grey image as well ?
    mean_file = '/imatge/asalvador/work/chalearn/models/bvlc_reference_caffenet/meanfile.npy'
    
    caffe.set_mode_gpu()
    net = caffe.Classifier(deploy_file, model_file, mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2,1,0),raw_scale=255)
    
    return net
    
def get_saliency(net,params):
    
    saliency_ratio = 0
    for i in np.arange(4)+1:
        
        QUERY_IMAGES = os.path.join(params['root'],'1_images','query' + params['year'])
        mask = cv2.imread( os.path.join( QUERY_IMAGES, params['query_name'], params['query_name'] +'.' + str(i) + '.mask.bmp' ) )[:,:,0]
        imagepath = os.path.join(params['root'],'1_images', 'query' + params['year'],params['query_name'],params['query_name']+'.'+ str(i)+'.src.bmp')
        
        ymin,xmin,ymax,xmax = find_coordinates(mask)
        
        scores = net.predict([caffe.io.load_image(imagepath)])
    
        feat = net.blobs['deconv1'].data
        feat = np.resize(feat,(576,768) )
        
        saliency_ratio = saliency_ratio + sum( sum( feat[ymin:ymax,xmin:xmax])) / ((ymax-ymin)*(xmax-ymin))
        

    
    return saliency_ratio/4


if __name__ == '__main__':
    
    net = init_net(params)
    queries = range (9099,9129)
    
    for query in queries:
        if query not in (9100,9113,9117):
            params['query_name'] = str(query)
            
            print get_saliency(net,params)

                       
