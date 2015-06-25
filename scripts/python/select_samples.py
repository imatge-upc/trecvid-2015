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
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'trecvid': ('trecvid',
                    'vgg_cnn_m_1024_fast_rcnn_trecvid_0_1_iter_40000.caffemodel' )}


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


    net = fast_rcnn_comp.init_net(params)

    queries = range(9099,9129)
    iou_overlap = 0.1
    iou_negatives = 0.1

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

            if not os.path.isdir(os.path.join(DESCRIPTORS_PATH,'train')):
                os.makedirs(os.path.join(DESCRIPTORS_PATH,'train'))
                os.makedirs(os.path.join(DESCRIPTORS_PATH,'test'))
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
                boxes = boxes[0:min(params['num_candidates'],np.shape(boxes)[0]),:]

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

                # Randomize
                '''
                negatives = random.sample(negatives, len(negatives))
                positives = random.sample(positives, len(positives))
                '''
                positives_train = positives[0:int(params['split_percentage']*len(positives))]
                positives_test = positives[int(params['split_percentage']*len(positives)):len(positives)]

                negatives_train = negatives[0:int(params['split_percentage']*len(negatives))]
                negatives_test = negatives[int(params['split_percentage']*len(negatives)):len(negatives)]

                print "Number of positives: ", len(positives), len(positives_train), len(positives_test)
                print "Number of negatives: ", len(negatives), len(negatives_train), len(negatives_test)


                boxes_train = boxes[positives_train + negatives_train,:]
                boxes_test = boxes[positives_test + negatives_test,:]

                labels_train = np.ones(len(boxes_train))
                labels_test = np.ones(len(boxes_test))

                labels_train[len(positives_train):len(labels_train)] = 0
                labels_test[len(positives_test):len(labels_test)] = 0


                if not os.path.isfile(os.path.join(DESCRIPTORS_PATH,'train',params['query_name']+'.' + str(i)+ '.src.bmp.p')):

                    #boxes = np.vstack( (np.array([xmin,ymin,xmax,ymax]), boxes))

                    file_to_save = open(os.path.join(DESCRIPTORS_PATH, 'train',params['query_name']+'.' + str(i)+ '.src.bmp.p'),'wb')

                    # Loop to extract features - due to memory problems


                    all_feats, _ = fast_rcnn_comp.extract_features(params,net, image_name,boxes_train)

                    pickle.dump(all_feats,file_to_save)
                    pickle.dump(boxes_train,file_to_save)
                    pickle.dump(labels_train,file_to_save)

                    file_to_save.close()

                    print "Train samples:", np.shape(all_feats)
                else:
                    print "Already stored. Skipping..."

                if not os.path.isfile(os.path.join(DESCRIPTORS_PATH,'test',params['query_name']+'.' + str(i)+ '.src.bmp.p')):

                    #boxes = np.vstack( (np.array([xmin,ymin,xmax,ymax]), boxes))

                    file_to_save = open(os.path.join(DESCRIPTORS_PATH,'test',params['query_name']+'.' + str(i)+ '.src.bmp.p'),'wb')

                    # Loop to extract features - due to memory problems

                    all_feats, _ = fast_rcnn_comp.extract_features(params,net, image_name,boxes_test)

                    pickle.dump(all_feats,file_to_save)
                    pickle.dump(boxes_test,file_to_save)
                    pickle.dump(labels_test,file_to_save)

                    file_to_save.close()

                    print " Test samples:", np.shape(all_feats)
                else:
                    print "Already stored. Skipping..."




