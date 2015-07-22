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
        
        if 'fullrank' in params['baseline']:
            BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],'all_frames.txt')
            with open(BASELINE_RANKING,'r') as f:
                shot_list = f.readlines()
        else:
            BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')
            shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
            
        if params['rerank_bool']:
            shot_list = shot_list[0:params['length_ranking']]

    DISTANCES_PATH = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'])
    
    
    i = 0
    shots = []
    frame_list = []
    distance_list = []
    region_list = []
    errors = []
    # For all my shots...
    for shot in shot_list:
        try:
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
                print shot, f
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
                
            
            frame_list.append(frame)
            region_list.append(region)
            
            if 'svm' in params['distance_type']:
                distance_list.append(distance[0])
            else:
                distance_list.append(distance)
            shots.append(shot)
        except:
            
            errors.append(shot)
            print "Could not merge for shot", shot
           
        i = i + 1

    # Ranking
    print errors, np.shape(errors)   
    RANKING_FILE = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['distance_type'])        
        
    if not os.path.isdir(RANKING_FILE):
        os.makedirs(RANKING_FILE)

    RANKING_FILE = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'] + '.rank')

    file_to_save = open(RANKING_FILE,'wb')
    
    if 'euclidean' in params['distance_type']:
        ranking = np.array(shots)[np.argsort(distance_list)]
        frames = np.array(frame_list)[np.argsort(distance_list)]
        regions = np.array(region_list)[np.argsort(distance_list)]
        distances = np.array(distance_list)[np.argsort(distance_list)]
    else:

        ranking = np.array(shots)[np.argsort(distance_list)[::-1]]
        frames = np.array(frame_list)[np.argsort(distance_list)[::-1]]
        regions = np.array(region_list)[np.argsort(distance_list)[::-1]]
        distances = np.array(distance_list)[np.argsort(distance_list)[::-1]]

    pickle.dump(ranking,file_to_save)
    pickle.dump(frames,file_to_save)
    pickle.dump(regions,file_to_save)
    pickle.dump(distances,file_to_save)
    pickle.dump(np.array(distance_list),file_to_save)
    file_to_save.close()
        
    # This removes frame distance files
    #shutil.rmtree(os.path.join(DISTANCES_PATH,shot))

def ranking_tv(params,eval_file):

    RANKING_FILE = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['query_name'] + '.rank')
    f = open(RANKING_FILE,'rb')
    ranking = pickle.load(f)
    f.close()


    i = 0
    for shot in ranking:

        eval_file.write(params['query_name'] + '\t' + '0' + '\t' + shot + '\t' + str(i) + '\t' + str(params['length_ranking'] - i) + '\t' + 'NII' +'\n'  )
        i = i + 1

    f.close()

if __name__ == '__main__':
    params = get_params()
    params['query_name'] = str(sys.argv[1])
    
    ts = time.time()
    if params['query_name'] not in (9100,9113,9117):
        merge_distances(params)
        print "Merged ranking for query", params['query_name'], 'in', time.time() - ts, 'seconds.'