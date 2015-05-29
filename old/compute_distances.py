import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import glob
import time
import sys

NUM_OUTPUT = 4096

DB_IMAGES = '/imatge/asalvador/work/trecvid/images_1000_4_nii/'
QUERY_IMAGES = '/imatge/asalvador/work/trecvid/query_images_2014/query2014/'
DB_DESCRIPTORS_PATH = '/imatge/asalvador/work/trecvid/descriptors/selective_search_descriptors2014/'
QUERY_DESCRIPTORS_PATH = '/imatge/asalvador/work/trecvid/descriptors/query_images_2014/'
QUERY_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_nii/'

SAVE_RCNN_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_rcnn'

SAVE_DISTANCES = '/imatge/asalvador/work/trecvid/distances/rcnn/'


def read_csv_df(csv_file):

    df = pd.read_csv(csv_file)

    # Define list of column names
    class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]

    region_cols = ['ymin','xmin','ymax','xmax']

    # Take values as feature matrix
    predictions = df[class_cols].values

    filenames = df['filename'].values

    # Take the region information for visualization
    regions = df[region_cols].values

    return predictions, regions, filenames

def frames_in_shot(shot_name,path_to_frames):

    frame_list = glob.glob( os.path.join(path_to_frames,shot_name) + '/*.jpg')

    return frame_list




if __name__ == '__main__':

    query = 9099
    path_to_save = os.path.join(SAVE_DISTANCES, str(query))

    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)

    num_errors = 0
    error_image_names = []

    # Use index to break the process
    i = int(float(sys.argv[1]))

    # Read the query descriptors
    features_query, regions_query, filenames = read_csv_df(os.path.join(QUERY_DESCRIPTORS_PATH,str(query) + '.csv'))


    # Load the results list that we want to rerank

    shot_list = pickle.load(open( os.path.join(QUERY_RANKINGS,str(query)) + '.rank', 'rb') )
    shot_list = shot_list[0:1000]

    shot = shot_list[i]



    # Lists to fill in
    frame_list = []
    region_list = []
    distance_list = []

    ts = time.time()
    if not os.path.isfile(os.path.join(path_to_save,shot + '.dist')):

        file_to_save = open(os.path.join(path_to_save,shot + '.dist'),'wb')
        # Get frame names in the shot
        images = frames_in_shot(shot,DB_IMAGES)

        shot_distances = []
        shot_regions = []

        # And for each one of them
        for im in images:

            # Get the name isolated
            frame = im.split('/')[len( im.split('/') )-1][:-4]

            # Read the database descriptors
            features_db, regions_db, filenames = read_csv_df( os.path.join(DB_DESCRIPTORS_PATH,shot + '/' + frame + '.csv') )

            try:
                # Compute distances of all regions to query regions
                distances = euclidean_distances(features_db,features_query)

                # Take the average distance to the 4 query regions
                distances =  np.mean(distances,axis=1)

                # Take the position of minimum distance
                idx = np.argmin(distances)

                # Save the region with minimum distance
                matching_region = regions_db[idx]

                # And keep the distance as well to use as score
                distance = np.min(distances)

            except:

                print "Error ! Features for image ", frame, "raised some error and could not be used."
                print "Saving this to log..."

                # Give a high value for distance so that this does not get picked as best frame
                distance = 1000000
                matching_region = [ 0, 0, 0, 0]

                error_image_names.append(frame)
                num_errors += 1
            #
            shot_distances.append(distance)
            shot_regions.append(matching_region)

            i += 1

        # Take the minimum distance
        idx = np.argmin(shot_distances)

        # Select the image that caused it
        frame = images[idx]

        # And the region within that image
        region = shot_regions[idx]

        # And the distance to the query:
        distance = np.min(shot_distances)

        print "Processed one shot in ", time.time() - ts, 'seconds.'

        print "Saving..."


        pickle.dump(frame,file_to_save)
        pickle.dump(region,file_to_save)
        pickle.dump(distance,file_to_save)

        file_to_save.close()
    # Sort shot list according to distances
    else:

        print "Distance for shot", shot, "already stored. Exiting..."

    '''
    ranking = np.array(shot_list)[np.argsort(distance_list)]
    frames = np.array(frame_list)[np.argsort(distance_list)]
    regions = np.array(region_list)[np.argsort(distance_list)]
    distances = np.array(distance_list)[np.argsort(distance_list)]


    if not os.path.isdir(SAVE_RCNN_RANKINGS):
        os.makedirs(SAVE_RCNN_RANKINGS)

    # Store results to file
    file_to_save = open( os.path.join(SAVE_RCNN_RANKINGS, str(query) + '.rank') ,'wb')

    pickle.dump(ranking,file_to_save)
    pickle.dump(frames,file_to_save)
    pickle.dump(regions,file_to_save)
    pickle.dump(distances,file_to_save)

    file_to_save.close()

    print "Done."
    print "Number of unprocessed frames: ", num_errors
    print "Name of unprocessed frames: "
    print error_image_names
    '''



