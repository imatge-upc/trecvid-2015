import pickle
import numpy as np
import os

PATH_TO_RCNN_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_rcnn'
PATH_TO_DISTANCES = '/imatge/asalvador/work/trecvid/distances/rcnn/'

def store_ranking(query):

    shot_list = os.listdir(os.path.join(PATH_TO_DISTANCES,str(query)))

    file_to_save = open(os.path.join(PATH_TO_RCNN_RANKINGS, str(query) + '.rank'),'wb')
    print "File is ready."
    frame_list = []
    region_list = []
    distance_list = []
    shots = []
    i = 0
    for shot in shot_list:
        print shot, i
        i = i + 1
        shot_info_file = open( os.path.join( os.path.join( PATH_TO_DISTANCES,str(query) ) , shot), 'rb')

        frame_list.append(pickle.load(shot_info_file))
        region_list.append(pickle.load(shot_info_file))
        distance_list.append(pickle.load(shot_info_file))
        shots.append(shot.split('.dist')[0])
        shot_info_file.close()

    ranking = np.array(shots)[np.argsort(distance_list)]
    frames = np.array(frame_list)[np.argsort(distance_list)]
    regions = np.array(region_list)[np.argsort(distance_list)]
    distances = np.array(distance_list)[np.argsort(distance_list)]

    pickle.dump(ranking,file_to_save)
    pickle.dump(frames,file_to_save)
    pickle.dump(regions,file_to_save)
    pickle.dump(distances,file_to_save)
    pickle.dump(np.array(distance_list),file_to_save)

    file_to_save.close()


if __name__ == "__main__":

    query = 9099

    store_ranking(query)

    print "Stored ranking for query", query, '.'