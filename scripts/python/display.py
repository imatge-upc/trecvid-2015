import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import evaluate as eval
from get_params import get_params
""" Run this to save figures displaying top 12 results of the ranking for each query."""
params = get_params()
# Image paths
DB_IMAGES = os.path.join(params['root'],'1_images/full')
QUERY_IMAGES = os.path.join(params['root'],'1_images/query2014')

RANKING_PATH = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'],params['distance_type'])

QUERY_DATA = os.path.join(params['root'], '4_object_proposals', 'query' + params['year'] + '_gt/csv')

GROUND_TRUTH_FILE = os.path.join(params['root'],'8_groundtruth','src','ins.search.qrels.tv14')

FIGURES_PATH = os.path.join(params['root'], '9_other/figures', params['distance_type'])

if not os.path.isdir(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

RED = [255,0,0]
GREEN = [0,255,0]


def display(params):

    ranking,db_frames,db_regions, query_images,query_regions, labels = get_data(params)

    print params['query_name']
    fig = plt.figure(figsize=(20,10))

    labels = labels[0]
    for i in range(len(query_images)):

        img_query = cv2.cvtColor( cv2.imread(query_images[i]), cv2.COLOR_BGR2RGB)

        cv2.rectangle(img_query, (int(query_regions[i][0]), int(query_regions[i][1])), (int(query_regions[i][2]), int(query_regions[i][3])), 255,5)

        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(img_query)

    for j in range(12):

        img_db = cv2.cvtColor( cv2.imread(os.path.join(DB_IMAGES,ranking[j],db_frames[j] + '.jpg')),  cv2.COLOR_BGR2RGB)

        cv2.rectangle(img_db, (int(db_regions[j][0]), int(db_regions[j][1])), (int(db_regions[j][2]), int(db_regions[j][3])), 255,5)

        if labels[j]==1:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=GREEN)
        else:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)
        ax = fig.add_subplot(4, 4, 5+j)
        ax.imshow(img_db)

    print "Displaying..."
    plt.axis('off')
    plt.savefig(os.path.join(FIGURES_PATH,params['query_name'] + '.png'))
    plt.close()
    #plt.show()

def get_data(params):

    # Ranking info

    f = open(os.path.join(RANKING_PATH,params['query_name'] + '.rank'))

    ranking = pickle.load(f)
    frames = pickle.load(f)
    regions = pickle.load(f)

    f.close()

    with(open(os.path.join(QUERY_DATA, params['query_name']+'.csv'),'r')) as f:
        image_list = f.readlines()

    query_images = []
    regions_query = []

    for pos in range( len(image_list) - 1):

        line_to_read = image_list[pos + 1]

        line_parts = line_to_read.split(',')



        ymin = int(float(line_parts[1]))
        xmin = int(float(line_parts[2]))
        ymax = int(float(line_parts[3]))
        xmax = int(float(line_parts[4]))


        boxes = np.array([xmin,ymin,xmax,ymax])

        if len(regions_query) == 0:
            regions_query = np.reshape(boxes,(1,4))
        else:
            regions_query = np.vstack((regions_query,np.reshape(boxes,(1,4))))

        query_images.append(line_parts[0])

    labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, params['query_name'], ranking)

    return ranking,frames,regions, query_images,regions_query,labels

if __name__ == '__main__':

    params = get_params()
    queries = range(9099,9129)

    for query in queries:

        if query not in [9100,9113,9117]:
            params['query_name'] = str(query)

            if os.path.isfile( os.path.join( RANKING_PATH,params['query_name'] + '.rank') ):

                display(params)



