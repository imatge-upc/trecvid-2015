from get_params import get_params
import sys
import cv2
import os
import scipy.io as sio
import argparse
import numpy as np
from sklearn.svm import SVC
import time
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import shutil

""" Feature extraction and distance computation """
params = get_params()

sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
import fast_rcnn.test as test_ops

classes = ('__background__', # always index 0
                         '9099', '9100','9101', '9102', '9103',
                         '9104', '9105', '9106', '9107', '9108',
                         '9109', '9110', '9111', '9112', '9113',
                         '9114', '9115', '9116', '9117',
                         '9118', '9119', '9120', '9121','9122',
                         '9123','9124','9125','9126','9127','9128')

class_to_ind = dict(zip(classes, xrange(len(classes))))

# This is taken from Fast-rcnn code
NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'trecvid': ('trecvid',
                    'vgg_cnn_m_1024_fast_rcnn_trecvid_iter_40000.caffemodel' )}




def extract_features(params,net, im_file, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """


    im = cv2.imread(im_file)
    blobs, unused_im_scale_factors = test_ops._get_blobs(im, boxes)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))

    if 'fc' in params['layer'] or 'score' in params['layer']:
        scores = net.blobs[params['layer']].data
    else:
        scores = blobs_out[params['layer']]

    return scores, boxes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def init_net(params):

    args = parse_args()

    if 'trecvid' in params['net_name']:
        prototxt = os.path.join(params['fast_rcnn_path'],'models', NETS[params['net_name']][0], 'test_' + params['net_name']+'.prototxt')
    else:
        prototxt = os.path.join(params['fast_rcnn_path'],'models', NETS[params['net_name']][0], 'test.prototxt')

    caffemodel = os.path.join(params['fast_rcnn_path'],'data', 'fast_rcnn_models',
                              NETS[params['net_name']][1])


    if params['gpu']:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    return net

def pw_distance(params,feats,query_feats,boxes,clf=None):

    if params['distance_type'] == 'euclidean':

        distances = pairwise_distances(feats,query_feats,params['distance_type'])

        # Take the average distance to the 4 query regions
        distances =  np.mean(distances,axis=1)

        # And the position
        idx = np.argmin(distances)

    elif 'svm' in params['distance_type']:

        distances = clf.decision_function(feats)

        idx = np.argmax(distances)

    else:
        # For fine tuned networks, where we directly have probabilities.
        i = class_to_ind[params['query_name']]
        distances = feats[:,i]
        idx = np.argmax(distances)

    # Save the region with minimum distance
    matching_region = boxes[idx]

    # And keep the distance as well to use as score
    distance = distances[idx]

    return distance,matching_region

def run(params,net,save_mode = False):

    if params['database'] == 'db' or params['database'] == 'full':

        DESCRIPTORS_PATH = os.path.join(params['root'], '5_descriptors',params['net'],params['database'] + params['year'])
        SELECTIVE_SEARCH_PATH = os.path.join(params['root'], '4_object_proposals', params['region_detector'], 'mat', params['database'] + params['year'])
        IMAGE_LIST = os.path.join(params['root'], '3_framelists', params['database'] + params['year'], params['query_name'] + '.txt')
        QUERY_DESCRIPTORS = os.path.join(params['root'], '5_descriptors',params['net'],'query' + params['year'],params['query_name'])
        DISTANCES_PATH = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'])

    elif params['database'] == 'query':

        DESCRIPTORS_PATH = os.path.join(params['root'], '5_descriptors',params['net'],params['database'] + params['year'])
        SELECTIVE_SEARCH_PATH = os.path.join(params['root'], '4_object_proposals', params['database'] + params['year'] + '_gt', 'csv',params['query_name'] + '.csv')

    if params['database'] =='db' or params['database'] == 'full':


        # Load query features

        if not save_mode:

            query_feats = []

            if 'svm' in params['distance_type']:
                #print os.path.join(params['root'], '9_other','svm_data', 'models',params['distance_type'],params['query_name'] + '.model')
                clf = pickle.load(open(os.path.join(params['root'], '9_other','svm_data', 'models',params['distance_type'],params['query_name'] + '.model'),'rb'))
            else:

                clf = None
                for f in os.listdir(QUERY_DESCRIPTORS):

                    if len(query_feats) == 0:

                        query_feats = pickle.load(open(os.path.join(QUERY_DESCRIPTORS,f),'rb'))

                    else:
                        query_feats = np.vstack((query_feats,pickle.load(open(os.path.join(QUERY_DESCRIPTORS,f),'rb'))))


        # Start for target database

        with(open(IMAGE_LIST,'r')) as f:
            image_list = f.readlines()


        i = 0
        for image_path in image_list:

            ts = time.time()
            i = i + 1

            # Get shot and image names
            image_name_parts = image_path.split('/')
            shot_name = image_name_parts[len(image_name_parts)-2]
            image_name = image_name_parts[len(image_name_parts)-1].rstrip()
           
            # The file to store features
            if save_mode:
                where_to_save = os.path.join(DESCRIPTORS_PATH, shot_name,image_name + '.p')
            else:
                where_to_save = os.path.join(DISTANCES_PATH,shot_name,image_name)

            # Only do this if we don't have the features yet
            if not os.path.isfile(where_to_save):

                # Create directories when necessary
                if not os.path.isdir(os.path.join(DESCRIPTORS_PATH, shot_name)):
                    os.makedirs(os.path.join(DESCRIPTORS_PATH, shot_name))

                # Load selective search boxes
                boxes_file = os.path.join(SELECTIVE_SEARCH_PATH,shot_name,image_name + '.mat')
                boxes = sio.loadmat(boxes_file)['boxes']
                
                boxes = boxes[0:min(params['num_candidates'],np.shape(boxes)[0]),:]
                
                # Extract features
                feats,boxes = extract_features(params,net, image_path.rstrip(), boxes)

                # We may want to save features to disk... but note that this is pretty slow compared to feature extraction
                if save_mode:

                    pickle.dump(feats,open(where_to_save,'wb'))

                else:

                    # So if we don't save features, we save distances instead

                    # Prepare the distance file
                    file_to_save = os.path.join(DISTANCES_PATH,shot_name,image_name)

                    # And if we don't have it yet...
                    if not os.path.isfile(file_to_save):

                        distance,matching_region = pw_distance(params,feats,query_feats,boxes,clf)

                        # Make new directories if necessary:
                        if not os.path.isdir(DISTANCES_PATH):
                            os.makedirs(DISTANCES_PATH)

                        if not os.path.isdir(os.path.join(DISTANCES_PATH,shot_name)):
                            os.makedirs(os.path.join(DISTANCES_PATH,shot_name))

                        # And dump score and best matching region
                        file_to_save = open(file_to_save,'wb')
                        pickle.dump(distance,file_to_save)
                        pickle.dump(matching_region,file_to_save)
                        file_to_save.close()
                    else:

                        print "Distance already available. Skipping..."

                print "Done for position",i, 'out of', len(image_list), 'in', time.time() - ts, 'seconds.'

            else:

                print "File already existed. Skipping..."


    elif params['database'] == 'query':

        # In this case we want features for the query regions, we can load those straight away...
        with(open(SELECTIVE_SEARCH_PATH,'r')) as f:
            image_list = f.readlines()

        # And for each one of them...
        for i in range(len(image_list) - 1 ):

            # Extract image path from the file lines
            q = image_list[i+1]
            image_name_parts = q.split(',')
            image_name = image_name_parts[0]

            # Extract the name from the image path
            path_parts = image_name.split('/')
            path_parts = path_parts[len(path_parts)-1]

            # This is the feature file to save
            where_to_save = os.path.join(DESCRIPTORS_PATH, params['query_name'],path_parts + '.p')

            # ...only if we don't have it yet
            if not os.path.isfile(where_to_save):

                # Make dirs...
                if not os.path.isdir(os.path.join(DESCRIPTORS_PATH, params['query_name'])):
                    os.makedirs(os.path.join(DESCRIPTORS_PATH, params['query_name']))

                # Get box information
                ymin = int(float(image_name_parts[1]))
                xmin = int(float(image_name_parts[2]))
                ymax = int(float(image_name_parts[3]))
                xmax = int(float(image_name_parts[4]))

                # And put it in a numpy array as if it were a selective search candidate
                boxes = np.array([xmin,ymin,xmax,ymax])
                boxes = np.reshape(boxes,(1,4))

                # Extract features
                feats,boxes = extract_features(params,net, image_name, boxes)

                # And save them
                if save_mode:
                    pickle.dump(feats,open(where_to_save,'wb'))

            else:
                print "File already existed. Skipping..."

def merge_distances(params):

    # This function takes the saved distances for each image and does max pooling for all images of the same shot.
    BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')
    DISTANCES_PATH = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'])

    shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
    shot_list = shot_list[0:params['length_ranking']]

    i = 0
    # For all my shots...
    for shot in shot_list:

        shot_files = os.listdir(os.path.join(DISTANCES_PATH,shot))

        shot_distances = []
        shot_regions = []
        images = []
        # Go through all frames in shot
        for f in shot_files:

            # Load the distances
            images.append(f[:-4])
            shot_info = open(os.path.join(DISTANCES_PATH,shot,f),'rb')

            distance = pickle.load(shot_info)
            matching_region = pickle.load(shot_info)
            shot_info.close()

            shot_distances.append(distance)
            shot_regions.append(matching_region)

        # Pooling over frames
        if params['distance_type'] == 'euclidean':
            # Take the minimum distance
            idx = np.argmin(shot_distances)
        else:
            idx = np.argmax(shot_distances)

        # Select the image that caused it
        frame = images[idx]

        # And the region within that image
        region = shot_regions[idx]

        # And the distance to the query:
        distance = shot_distances[idx]
        print distance
        file_to_save = open(os.path.join(DISTANCES_PATH,shot + '.dist'),'wb')

        # And we save this for the ranking
        pickle.dump(frame,file_to_save)
        pickle.dump(region,file_to_save)
        pickle.dump(distance,file_to_save)

        file_to_save.close()
        print "Merged for shot", i
        i = i + 1

        # This removes frame distance files
        shutil.rmtree(os.path.join(DISTANCES_PATH,shot))


if __name__ == '__main__':

    params = get_params()
    net = init_net(params)

    # Step 2. Extract features and compute distances
    params['database'] = 'full'

    queries = range(9099,9129)
    for query in queries:

        if query not in (9100,9113,9117):
            print query
            params['query_name'] = str(query)

            # This part extracts features and saves distances per frame
            run(params,net)
            # This part loads distance per frame and returns pooled distances per shot
            merge_distances(params)

    # To extract query features you need to run run() with save_mode == True and params['database'] = query



