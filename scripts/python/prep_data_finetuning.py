import numpy as np
import os
from get_params import get_params
import csv
params = get_params()
import pickle
""" Prepares query data for fine tuning fast-rcnn"""
imagesets = os.path.join(params['root'],'1_images','trecvid','imagesets')
annotations = os.path.join(params['root'],'1_images','trecvid','annotations')
split_ratio = 1
def image_sets(queries):

    val_list = []
    train_list = []

    for query in queries:

        for i in np.arange(4)+1:

            image_name =  str(query) +'.' + str(i) + '.src'

            if i > split_ratio*4:

                val_list.append(image_name)

            else:
                train_list.append(image_name)

    val_file = os.path.join(imagesets,'val.txt')
    train_file = os.path.join(imagesets,'train.txt')

    if not len(val_list) == 0:

        with open(val_file,'w') as f:
            f.write("\n".join(val_list))

    if not len(train_list) == 0:

        with open(train_file,'w') as f:
            f.write("\n".join(train_list))
    print len(train_list)
    '''
    i = 0

    for query in queries:

        labels = np.ones((np.shape(val_list)[0]))*(-1)
        labels[2*i:2*(i+1)] = 1
        labels = labels.astype(int)

        val_file = os.path.join(imagesets,str(query) + '_val.txt')
        train_file = os.path.join(imagesets,str(query) + '_train.txt')

        with open(val_file,'w') as f:

            writer = csv.writer(f,delimiter = ' ')
            writer.writerows(zip(val_list,labels))


        with open(train_file,'w') as f:

            writer = csv.writer(f,delimiter = ' ')
            writer.writerows(zip(train_list,labels))

        i = i + 1
    '''
def annotate(queries):

    for query in queries:

        annotations_file = os.path.join(params['root'],'4_object_proposals','query' + params['year'] + '_gt', 'csv', str(query) + '.csv')

        with open(annotations_file,'r') as a:
            lines = a.readlines()

        for i in range(len(lines) - 1 ):

            line = lines[i+1]
            image_name_parts = line.split(',')
            image_name =  str(query) +'.' + str(i+1) + '.src'

            f = open(os.path.join(annotations,image_name + '.xml'),'wb')

            ymin = int(float(image_name_parts[1]))
            xmin = int(float(image_name_parts[2]))
            ymax = int(float(image_name_parts[3]))
            xmax = int(float(image_name_parts[4]))

            print image_name,ymin,xmin,ymax,xmax
            pickle.dump(ymin,f)
            pickle.dump(xmin,f)
            pickle.dump(ymax,f)
            pickle.dump(xmax,f)
            pickle.dump(image_name,f)

            f.close()


if __name__ == "__main__":

    queries = range(9099,9129)

    image_sets(queries)

    annotate(queries)




