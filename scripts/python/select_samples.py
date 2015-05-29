import os
import numpy as np
import Image
import cv2
import matplotlib.pyplot as plt
import csv
import sys
import pickle
import scipy.io as sio
from get_params import get_params
import fast_rcnn_comp as fast_rcnn_comp
import random

params = get_params()

sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
import fast_rcnn.test as test_ops

# This is taken from Fast-rcnn code
NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def find_coordinates(mask):
    array_of_locations = np.where(mask==255)

    y = array_of_locations[0]
    x = array_of_locations[1]

    ymin = np.min(y)
    xmin = np.min(x)
    ymax = np.max(y)
    xmax = np.max(x)

    return ymin,xmin,ymax,xmax

def overlap(r1,r2):

    hoverlaps = True
    voverlaps = True

    if (r1[1] > r2[3]) or (r1[3] < r2[1]):
        voverlaps = False
    if (r1[0] > r2[2]) or (r1[2] < r2[0]):
        hoverlaps = False

    if hoverlaps and voverlaps:

        return True
    else:
        return False

def intersection_over_union(r1,r2):

    'xmin,ymin,xmax,ymax'
    size_candidate = r2[3] - r2[1] + r2[2] - r2[0]

    if overlap(r1,r2) :

        xmin_int = max(r1[0],r2[0])
        ymin_int = max(r1[1],r2[1])
        xmax_int = min(r1[2],r2[2])
        ymax_int = min(r1[3],r2[3])

        intersection = xmax_int - xmin_int + ymax_int - ymin_int
        union = size_candidate + region_size - intersection
    else:

        intersection = 0
        union = 1

    return float(intersection)/float(union)

def display_regions(image_object,regions):

    fig = plt.figure(figsize=(20,10))

    for j in range(min(16,np.shape(regions)[0])):
        img_db = cv2.cvtColor( image_object,  cv2.COLOR_BGR2RGB)
        cv2.rectangle(img_db, (int(regions[j][0]), int(regions[j][1])), (int(regions[j][2]), int(regions[j][3])), 255,5)


        ax = fig.add_subplot(4, 4, 1+j)
        ax.imshow(img_db)

    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    params = get_params()

    QUERY_IMAGES = os.path.join(params['root'],'1_images','query' + params['year'])
    SEL_SEARCH_PATH = os.path.join(params['root'],'4_object_proposals','selective_search','mat','query'+params['year'])
    SVM_DATA = os.path.join(params['root'],'9_other','svm_data')


    net = fast_rcnn_comp.init_net()

    queries = range(9099,9129)
    iou_overlap = 0.3
    iou_negatives = 0.0

    display_bool = False

    numfeats = 0
    for query in queries:

        if query not in (9100,9113,9117):
            print "===================="
            print query

            params['query_name'] = str(query)
            DESCRIPTORS_PATH = os.path.join(params['root'],'5_descriptors', params['net'], 'query' + params['year'] +'_selective_search', params['query_name'])

            if not os.path.isdir(DESCRIPTORS_PATH):
                os.makedirs(DESCRIPTORS_PATH)
            # 4 query examples

            for i in np.arange(4)+1:

                mask = cv2.imread( os.path.join( QUERY_IMAGES, params['query_name'], params['query_name'] +'.' + str(i) + '.mask.bmp' ) )[:,:,0]

                ymin, xmin, ymax, xmax = find_coordinates(mask)

                region_size = ymax - ymin + xmax - xmin

                # Read the image
                image_name =  os.path.join( QUERY_IMAGES, params['query_name'], params['query_name'] +'.' + str(i) + '.src.bmp' )
                image_object =  cv2.imread(image_name)


                # Selective search
                mat_file = os.path.join(SEL_SEARCH_PATH,params['query_name'],params['query_name']+'.' + str(i)+ '.src.bmp.mat')

                boxes = sio.loadmat(mat_file)['boxes']


                positives = []
                negatives = []

                ii = 0 # Keep track of positions

                for box in boxes:

                    # Find the intersection over union

                    iou = intersection_over_union([xmin,ymin,xmax,ymax],box)

                    if iou > iou_overlap:

                        positives.append(ii)
                    elif iou <= iou_negatives:

                        negatives.append(ii)

                    ii +=1



                if display_bool:
                    display_regions(image_object,boxes[positives])


                num_negatives = len(positives)*params['min_negatives']

                negatives = random.sample(negatives, min(len(negatives),num_negatives))

                print "Number of positives: ", len(positives)
                print "Number of negatives: ", len(negatives)

                numfeats = numfeats + len(positives) + len(negatives)
                boxes = boxes[positives + negatives,:]

                labels = np.ones(len(boxes)+1)

                labels[len(positives)+1:len(labels)] = 0


                if not os.path.isfile(os.path.join(DESCRIPTORS_PATH,params['query_name']+'.' + str(i)+ '.src.bmp.p')):

                    boxes = np.vstack( (np.array([xmin,ymin,xmax,ymax]), boxes))

                    file_to_save = open(os.path.join(DESCRIPTORS_PATH,params['query_name']+'.' + str(i)+ '.src.bmp.p'),'wb')

                    # Loop to extract features - due to memory problems
                    all_feats = []
                    for idx in range(0,len(boxes),params['batch_size']):
              
                        feats, _ = fast_rcnn_comp.extract_features(params,net, image_name,boxes[idx:min(len(boxes),idx+params['batch_size']),:])

                        if len(all_feats) == 0:

                            all_feats = feats

                        else:

                            all_feats = np.vstack((all_feats,feats))

                    pickle.dump(all_feats,file_to_save)
                    pickle.dump(boxes,file_to_save)
                    pickle.dump(labels,file_to_save)

                    print "Stored", np.shape(all_feats)[0], 'features for image', params['query_name'], i

                    file_to_save.close()
                else:
                    print "Already stored. Skipping..."



                print "Done"
    print "Total number of features:",numfeats



