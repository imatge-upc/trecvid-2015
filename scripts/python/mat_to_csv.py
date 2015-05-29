from get_params import get_params
import scipy.io
import os
import numpy as np
import csv

params = get_params()

if params['database'] == 'db' or params['database'] =='full':
    IMAGE_PATH =  params['root'] + '1_images/' + params['database']
else:
    IMAGE_PATH =  params['root'] + '1_images/' + params['database'] + params['year']

MAT_PATH = params['root'] + '4_object_proposals/' + params['region_detector'] + '/mat/' + params['database'] + params['year']
CSV_PATH = params['root'] + '4_object_proposals/' + params['region_detector'] + '/csv/' + params['database'] + params['year']
IMAGE_LIST = params['root'] + '3_framelists/' + params['database'] + params['year'] + '/' + params['query_name']+'.txt'

def mat_to_np(mat_file,key):

    mat = scipy.io.loadmat(mat_file)

    return mat[key]

def to_csv(boxes,image_name,csv_file):


    names = [image_name] * len(boxes)
    names = np.reshape(names,(np.shape(names)[0],1))

    header_ = np.array(['filename', 'ymin', 'xmin', 'ymax', 'xmax'])

    to_write = np.hstack((names,boxes))

    to_write =  np.vstack((header_,to_write))

    with open(csv_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerows(to_write)


if __name__ == '__main__':

    key = 'boxes'

    with open(IMAGE_LIST) as f:
        image_list = f.readlines()

    i = 1
    errors = []
    for im in image_list:

        shot = im.split('/')[len(im.split('/'))-2].rstrip()
        frame = im.split('/')[len(im.split('/'))-1].rstrip()

        mat_file = os.path.join(MAT_PATH,shot + '/' + frame + '.mat')

        if not os.path.isdir(CSV_PATH):
            os.makedirs(CSV_PATH)

        if not os.path.isdir(os.path.join(CSV_PATH,shot)):
            os.makedirs(os.path.join(CSV_PATH,shot))

        csv_file = os.path.join(CSV_PATH,shot,frame[:-4] + '.csv')


        image_file = os.path.join(IMAGE_PATH,shot,frame)

        if not os.path.isfile(csv_file):

            try:

                boxes = mat_to_np(mat_file,key)
                
                boxes = boxes[0:min(params['num_candidates'],np.shape(boxes)[0]),:]
            
                to_csv(boxes,image_file,csv_file)
                print "Success. Stored csv for position", i , "of the image list."

            except:

                print "Selective search for position", i, "could not be saved !"
                errors.append(i)

        else:
            print "File already existed"

        i = i + 1
        


    print errors