import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from compute_distances import read_csv_df
import pickle
import time
import evaluation as eval
import random

DB_IMAGES = '/imatge/asalvador/work/trecvid/images_100_4_nii/'
QUERY_IMAGES = '/imatge/asalvador/work/trecvid/query_images_2014/'
DB_DESCRIPTORS_PATH = '/imatge/asalvador/work/trecvid/descriptors/selective_search_descriptors2014/'
QUERY_DESCRIPTORS_PATH = '/imatge/asalvador/work/trecvid/descriptors/query_images_2014/'
QUERY_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_nii/'

SAVE_RCNN_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_rcnn'
BASELINE_RANKINGS = '/imatge/asalvador/work/trecvid/rankings2014_nii'

GROUND_TRUTH_FILE = '/imatge/asalvador/work/trecvid/ground_truth_files/ins.search.qrels.tv14'

RED = [255,0,0]
GREEN = [0,255,0]

def display_rcnn(query_images,query_regions,db_frames,db_regions,labels):

    fig = plt.figure(figsize=(20,10))

    for i in range(len(query_images)):

        img_query = cv2.cvtColor( cv2.imread(query_images[i]), cv2.COLOR_BGR2RGB)

        cv2.rectangle(img_query, (int(query_regions[i][1]), int(query_regions[i][0])), (int(query_regions[i][3]), int(query_regions[i][2])), 255,5)

        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(img_query)


    for j in range(12):

        img_db = cv2.cvtColor( cv2.imread(db_frames[j]),  cv2.COLOR_BGR2RGB)

        cv2.rectangle(img_db, (int(db_regions[j][1]), int(db_regions[j][0])), (int(db_regions[j][3]), int(db_regions[j][2])), 255,5)

        if labels[j]==1:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=GREEN)
        else:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)
        ax = fig.add_subplot(4, 4, 5+j)
        ax.imshow(img_db)

    print "Displaying..."
    plt.axis('off')
    plt.show()

def display_baseline(query_images,db_shots,labels):

    fig = plt.figure(figsize=(20,10))

    for i in range(len(query_images)):

        img_query = cv2.cvtColor( cv2.imread(query_images[i]), cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(img_query)



    for j in range(12):

        db_frame = random.choice(os.listdir(DB_IMAGES + db_shots[j]))

        print db_frame
        image_path = DB_IMAGES + db_shots[j] + '/' + db_frame
        print image_path
        img_db = cv2.cvtColor( cv2.imread(image_path),  cv2.COLOR_BGR2RGB)

        if labels[j]==1:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=GREEN)
        else:
             img_db= cv2.copyMakeBorder(img_db,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)

        ax = fig.add_subplot(4, 4, 5+j)
        ax.imshow(img_db)

    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    queries = [9099]
    rcnn = False
    display_bool = False
    _map = False

    fusion_bool = True
    if rcnn:
        for query in queries:

            f = open(os.path.join(SAVE_RCNN_RANKINGS,str(query) + '.rank'))

            ranking = pickle.load(f)
            frames = pickle.load(f)
            regions = pickle.load(f)
            distances = pickle.load(f)

            f.close()

            features_query, regions_query, query_filenames = read_csv_df(os.path.join(QUERY_DESCRIPTORS_PATH,str(query) + '.csv'))
            labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), ranking)

            if display_bool:
                display_rcnn(query_filenames,regions_query,frames,regions,np.squeeze(labels))

            if _map:

                ap = eval.AveragePrecision(np.squeeze(labels),num_relevant)
                print num_relevant, sum(sum(labels))
                print "R-CNN Average precision for query ", query, ':', ap
                print "R-CNN -in-subset- Average precision for query", query, ':', ap*num_relevant/sum(sum(labels))
    else:

        for query in queries:

            f = open(os.path.join(BASELINE_RANKINGS,str(query) + '.rank'))

            ranking = pickle.load(f)

            f.close()


            features_query, regions_query, query_filenames = read_csv_df(os.path.join(QUERY_DESCRIPTORS_PATH,str(query) + '.csv'))
            labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), ranking[0:1000])

            if display_bool:
                display_baseline(query_filenames, ranking,np.squeeze(labels))


            if _map:
                labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), ranking[0:1000])
                ap = eval.AveragePrecision(np.squeeze(labels),num_relevant)

                print num_relevant, sum(sum(labels))

                print "Baseline Average precision for query ", query, ':', ap

                print "Baseline -in-subset- Average precision for query", query, ':', ap*num_relevant/sum(sum(labels))

    if fusion_bool:

        alpha = 0.5
        f = open(os.path.join(BASELINE_RANKINGS,str(query) + '.rank'))

        ranking_bow = pickle.load(f)[0:1000]
        weights = pickle.load(f)[0:1000]
        f.close()
        weights = np.array(weights)


        weights = (weights - np.min(weights) ) /( np.max(weights) - np.min(weights) )

        f = open(os.path.join(SAVE_RCNN_RANKINGS,str(query) + '.rank'))

        ranking_rcnn = pickle.load(f)
        frames = pickle.load(f)
        regions = pickle.load(f)
        distances = pickle.load(f)
        original_distance = pickle.load(f)

        # From 0 to 1
        original_distance = (original_distance - np.min(original_distance) ) / (np.max(original_distance) - np.min(original_distance))


        new_scoring = (alpha*original_distance + (1-alpha)*weights)/2

        new_ranking = np.array(ranking_bow)[np.argsort(new_scoring)[::-1]]

        labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), new_ranking)
        ap = eval.AveragePrecision(np.squeeze(labels),num_relevant)

        print num_relevant, sum(sum(labels))

        print "Baseline Average precision for query ", query, ':', ap

        print "Baseline -in-subset- Average precision for query", query, ':', ap*num_relevant/sum(sum(labels))
