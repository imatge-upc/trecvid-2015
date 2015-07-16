from get_params import get_params
import pickle
import time
import sys
import os
import numpy as np

def merge_distances(params):

    # This function takes the saved distances for each image and does max pooling for all images of the same shot.

    if params['database'] =='gt_imgs':
        BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['database'],'all_frames' + '.txt')
        with open(BASELINE_RANKING,'r') as f:
            shot_list = f.readlines()
    else:
        BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')
        shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
        if not params['database'] == 'gt_imgs':
            shot_list = shot_list[0:params['length_ranking']]

    DISTANCES_PATH = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'])

    i = 0
    # For all my shots...
    for shot in shot_list:

        shot = shot.rstrip()
        shot_files = os.listdir(os.path.join(DISTANCES_PATH,shot))

        shot_distances = []
        shot_regions = []
        images = []
        
        # Go through all frames in shot
        for f in shot_files:
            # Load the distances
            images.append(f[:-4])
            shot_info = open(os.path.join(DISTANCES_PATH,shot,f),'rb')

            distances = pickle.load(shot_info)
            matching_regions = pickle.load(shot_info)
            
            
            if params['database'] == 'gt_imgs':
                pos = int(float(params['query_name'])) - 9069
            else:
                pos = int(float(params['query_name'])) - 9099

            if 'scores' in params['distance_type']:
                pos = pos + 1 # there are 31 classes in the class score layer, and 0 is the background
            
            matching_region = matching_regions[pos,pos*4:pos*4+4]
            
            shot_distances.append(distances[pos])
            shot_regions.append(matching_region)

        # Pooling over frames
        if 'euclidean' in params['distance_type']:
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
        
        if not os.path.isdir(os.path.join(DISTANCES_PATH,params['query_name'])):
            os.makedirs(os.path.join(DISTANCES_PATH,params['query_name']))
            
        file_to_save = open(os.path.join(DISTANCES_PATH,params['query_name'],shot + '.dist'),'wb')

        # And we save this for the ranking
        pickle.dump(frame,file_to_save)
        pickle.dump(region,file_to_save)
        pickle.dump(distance,file_to_save)

        file_to_save.close()
        i = i + 1

        # This removes frame distance files
        #shutil.rmtree(os.path.join(DISTANCES_PATH,shot))

if __name__ == '__main__':
    params = get_params()
    params['query_name'] = str(sys.argv[1])
    
    ts = time.time()
    if params['query_name'] not in (9100,9113,9117):
        merge_distances(params)
        print "Merged for query", params['query_name'], 'in', time.time() - ts, 'seconds.'