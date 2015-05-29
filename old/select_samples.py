import os
import numpy as np
import Image
import cv2
import matplotlib.pyplot as plt
import csv
import sys
import pickle
selective_search_path = '/imatge/asalvador/workspace/selective_search_ijcv_with_python'
sys.path.insert(0,selective_search_path)

import selective_search


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
        hoverlaps = False
    if (r1[0] > r2[2]) or (r1[2] < r2[0]):
        voverlaps = False

    if hoverlaps and voverlaps:

        return True
    else:
        return False

def intersection_over_union(r1,r2):

    size_candidate = r2[3] - r2[1] + r2[2] - r2[0]

    if overlap(r1,r2) :

        ymin_int = max(r1[0],r2[0])
        xmin_int = max(r1[1],r2[1])
        ymax_int = min(r1[2],r2[2])
        xmax_int = min(r1[3],r2[3])

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
        cv2.rectangle(img_db, (int(regions[j][1]), int(regions[j][0])), (int(regions[j][3]), int(regions[j][2])), 255,5)


        ax = fig.add_subplot(4, 4, 1+j)
        ax.imshow(img_db)

    plt.axis('off')
    plt.show()


QUERY_IMAGES = '/imatge/asalvador/work/trecvid/query_images_2014/query2014/'
SEL_SEARCH_PATH = '/imatge/asalvador/work/trecvid/object_proposals/selective_search_queries2014/'
SVM_DATA = '/imatge/asalvador/work/trecvid/svm_data/query2014/'

queries = range(9099,9129)
iou_overlap = 0.5 # According to RCNN paper (finetuning)
display_bool = False

for query in queries:
    print "===================="
    print query

    # 4 query examples

    for i in np.arange(4)+1:

        mask = cv2.imread(QUERY_IMAGES + str(query) + '/' + str(query)+'.' + str(i) + '.mask.bmp')[:,:,0]

        ymin, xmin, ymax, xmax = find_coordinates(mask)

        region_size = ymax - ymin + xmax - xmin

        # Read the image
        image_name = QUERY_IMAGES + str(query) + '/' + str(query)+'.' + str(i) + '.src.bmp'
        image_object =  cv2.imread(image_name)

        # Selective search
        csv_file = os.path.join(SEL_SEARCH_PATH,str(query)+'.' + str(i)+'.csv')
        boxes = selective_search.get_windows([QUERY_IMAGES + str(query) + '/' + str(query)+'.' + str(i) + '.src.bmp'],cmd='selective_search')
        boxes = np.squeeze(boxes)

        # Save regions
        image_array_for_csv = [image_name] * len(boxes)
        image_array_for_csv = np.reshape( image_array_for_csv,((len(image_array_for_csv),1))  )

        selective_search.to_csv(boxes,image_array_for_csv,csv_file)

        positives = []
        negatives = []

        ii = 0 # Keep track of positions

        for box in boxes:

            # Find the intersection over union

            iou = intersection_over_union([ymin,xmin,ymax,xmax],box)

            if iou > iou_overlap:

                positives.append(ii)
            else:

                negatives.append(ii)

            ii +=1


        print "Number of positives: ", len(positives)
        # Display only 1 out of 4
        if display_bool:
            display_regions(image_object,boxes[positives])

        file_to_save = open(os.path.join(SVM_DATA,str(query)+'.' + str(i)+'.p'),'wb')
        pickle.dump(positives,file_to_save)
        pickle.dump(negatives,file_to_save)

        file_to_save.close()
        # Classify between positive and negative



