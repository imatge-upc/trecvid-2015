import os
import numpy as np
import Image
import cv2
import matplotlib.pyplot as plt
import csv
from get_params import get_params
""" Prepares query data for SVM """
params = get_params()

QUERY_IMAGES = os.path.join(params['root'], '1_images', 'query' + params['year'])

if params['year'] == '2015':
    queries = range(9129,9159)
elif params['year'] =='2014':
    queries = range(9099,9129)

where_to_save = os.path.join(params['root'], '4_object_proposals', 'query' + params['year'] + '_gt','csv')

if not os.path.isdir(where_to_save):

    os.makedirs(where_to_save)

for query in queries:

    lines_to_write = []

    for i in np.arange(4)+1:
        print QUERY_IMAGES + '/'+ str(query) + '/' + str(query)+'.' + str(i) + '.mask.bmp'
        mask = cv2.imread(QUERY_IMAGES + '/'+ str(query) + '/' + str(query)+'.' + str(i) + '.mask.bmp')[:,:,0]

        #image =  cv2.imread(QUERY_IMAGES + str(query) + '/' + str(query)+'.' + str(i) + '.src.bmp')

        array_of_locations = np.where(mask==255)

        y = array_of_locations[0]
        x = array_of_locations[1]

        image_name = QUERY_IMAGES + '/'+ str(query) + '/' + str(query)+'.' + str(i) + '.src.bmp'

        xmax = np.max(x)
        xmin = np.min(x)
        ymax = np.max(y)
        ymin = np.min(y)

        lines_to_write.append([image_name, ymin,xmin,ymax,xmax])

    file_to_save = os.path.join(where_to_save,str(query) + '.csv')

    header_ = np.array(['filename', 'ymin', 'xmin', 'ymax', 'xmax'])
    to_write =  np.vstack((header_,lines_to_write))
    
    with open(file_to_save, 'wb') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerows(to_write)

        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), 255)
