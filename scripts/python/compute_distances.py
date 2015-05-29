from get_params import get_params
import os
import convert_feats
import sys
import pickle
import numpy as np
import glob
from sklearn.metrics.pairwise import euclidean_distances
from mat_to_csv import mat_to_np
import time
""" OLD: Compute distances for extracted descriptors to query. This function is not longer used, distance computation is performed right after feature extraction. """
params = get_params()

DISTANCES_PATH  = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['query_name'])
FEAT_PATH = params['root'] + '5_descriptors/' + params['net'] + '/' + params['database'] + params['year']
MAT_PATH = params['root'] + '4_object_proposals/' + params['region_detector'] + '/mat/' + params['database'] + params['year']

QUERY_FEAT_PATH = os.path.join(params['root'],'5_descriptors',params['net'],'query' + params['year'],params['query_name'])
BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')

if params['database'] == 'db' or params['database'] =='full' :
    IMAGE_PATH =  params['root'] + '1_images/' + params['database']
else:
    IMAGE_PATH =  params['root'] + '1_images/' + params['database'] + params['year']


def frames_in_shot(shot_name,path_to_frames):

    frame_list = glob.glob( os.path.join(path_to_frames,shot_name) + '/*.jpg')

    return frame_list

if not os.path.isdir(DISTANCES_PATH):
    os.makedirs(DISTANCES_PATH)

if __name__ == '__main__':

    i = int(float(sys.argv[1]))

    shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
    shot_list = shot_list[0:params['length_ranking']]

    shot = shot_list[i]

    if not os.path.isfile(os.path.join(DISTANCES_PATH,shot + '.dist')):

            file_to_save = open(os.path.join(DISTANCES_PATH,shot + '.dist'),'wb')

            # Get frame names in the shot
            images = frames_in_shot(shot,IMAGE_PATH)

            shot_distances = []
            shot_regions = []

            # Loading query feats

            if params['net'] == 'spp':

                predictions_query = []

                for matfile in os.listdir(QUERY_FEAT_PATH):

                    query_feat_file = os.path.join(QUERY_FEAT_PATH,matfile)
                    predictions_query.append(mat_to_np(query_feat_file,'feats'))

                predictions_query = np.squeeze(np.array(predictions_query))

            else:
                predictions_query , regions_query = convert_feats.read_csv_feats(QUERY_FEAT_PATH + '.csv')



            # And for each one of them

            for im in images:

                # Get the name isolated

                frame = im.split('/')[len( im.split('/') )-1][:-4]

                if params['net'] == 'spp':


                    mat_file = os.path.join(MAT_PATH,shot,frame+'.jpg.mat')

                    predictions_file = os.path.join(FEAT_PATH,shot,frame+'.jpg.mat')

                    predictions,regions = convert_feats.read_mat_feats(predictions_file,mat_file)
                    regions = regions[0:min(params['num_candidates'],np.shape(regions)[0]),:]
                    predictions = predictions[0:min(params['num_candidates'],np.shape(predictions)[0]),:]


                else:


                    predictions_file = os.path.join(FEAT_PATH,shot,frame+'.csv')
                    predictions,regions = convert_feats.read_csv_feats(predictions_file)
                    
                    regions = regions[0:min(params['num_candidates'],np.shape(regions)[0]),:]
                    predictions = predictions[0:min(params['num_candidates'],np.shape(predictions)[0]),:]

                if params['delete_mode']:
                    os.remove(predictions_file)

                try:
                    # Compute distances of all regions to query regions
                    distances = euclidean_distances(predictions,predictions_query)

                    # Take the average distance to the 4 query regions
                    distances =  np.mean(distances,axis=1)

                    # Take the position of minimum distance
                    idx = np.argmin(distances)

                    # Save the region with minimum distance
                    matching_region = regions[idx]

                    # And keep the distance as well to use as score
                    distance = np.min(distances)


                except:

                    print "Error ! Features for image ", frame, "raised some error and could not be used."
                    print "Array position:", i

                    # Give a high value for distance so that this does not get picked as best frame
                    distance = 1000000
                    matching_region = [ 0, 0, 0, 0]


                shot_distances.append(distance)
                shot_regions.append(matching_region)


            # Take the minimum distance
            idx = np.argmin(shot_distances)

            # Select the image that caused it
            frame = images[idx]

            # And the region within that image
            region = shot_regions[idx]

            # And the distance to the query:
            distance = np.min(shot_distances)

            print "Saving..."


            pickle.dump(frame,file_to_save)
            pickle.dump(region,file_to_save)
            pickle.dump(distance,file_to_save)

            file_to_save.close()
        # Sort shot list according to distances
    else:

        print "Distance for shot", shot, "was already stored. Exiting..."
