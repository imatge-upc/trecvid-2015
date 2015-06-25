import pickle
import numpy as np
import os
import sys
import shutil
from get_params import get_params

""" Run this to generate rankings by reading all computed distances and sorting. """
def store_ranking(params):
    if params['database'] == 'gt_imgs':
        DISTANCES_PATH  = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'])
        BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['database'],'all_frames' + '.txt')

    else:
            
        DISTANCES_PATH  = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'])
        BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')

    RANKING_FILE = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['distance_type'])

    if not os.path.isdir(RANKING_FILE):
        os.makedirs(RANKING_FILE)

    RANKING_FILE = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'] + '.rank')

    if params['database'] == 'gt_imgs':
        with open(BASELINE_RANKING,'r') as f:
            shot_list = f.readlines()
    else:
        shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
        shot_list = shot_list[0:params['length_ranking']]

    file_to_save = open(RANKING_FILE,'wb')


    print "File is ready."

    frame_list = []
    region_list = []
    distance_list = []
    shots = []
    i = 0
    for shot in shot_list:
        print shot, i
        shot = shot.rstrip()
        i = i + 1
        shot_info_file = open( os.path.join(DISTANCES_PATH , shot + '.dist'), 'rb')

        frame_list.append(pickle.load(shot_info_file))
        region_list.append(pickle.load(shot_info_file))
        if 'svm' in params['distance_type']:
            dist = pickle.load(shot_info_file)
            distance_list.append(dist[0])
        else:
            distance_list.append(pickle.load(shot_info_file))
        shots.append(shot.split('.dist')[0])


        shot_info_file.close()



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
if __name__ == "__main__":

    params = get_params()
    '''
    if params['year'] =='2013':
        queries = range(9069,9099)
    else:
        queries = range(9099,9129)
    '''
    query = sys.argv[1]
    if query not in (9100,9113,9117):
        params['query_name'] = str(query)

        DISTANCES_PATH  = os.path.join(params['root'], '6_distances',params['net'],params['database'] + params['year'],params['distance_type'],params['query_name'])

        if os.path.isdir(DISTANCES_PATH):
            store_ranking(params)

            print "Stored ranking for query", params['query_name']
            #shutil.rmtree(DISTANCES_PATH)
        else:
            print "Query", params['query_name'], "not processed."

    
