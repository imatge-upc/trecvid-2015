from get_params import get_params
import os
import numpy as np
import pandas as pd
from mat_to_csv import mat_to_np

""" OLD: Used to transform RCNN or SPP_net features into a readable format in python. With fast-rcnn features are no longer stored, so this function is not used."""
NUM_OUTPUT = 4096
params = get_params()

FEAT_PATH = params['root'] + '5_descriptors/' + params['net'] + '/' + params['database']
MAT_PATH = params['root'] + '4_object_proposals/' + params['region_detector'] + '/mat/' + params['database'] + params['year']

QUERY_FEAT_PATH = os.path.join(params['root'],'5_descriptors',params['net'],'query' + params['year'],params['query_name'])

def read_csv_feats(csv_file):

    # With csv features (R-CNN)

    df = pd.read_csv(csv_file)

    # Define list of column names
    class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]

    region_cols = ['ymin','xmin','ymax','xmax']

    # Take values as feature matrix
    predictions = df[class_cols].values

    # Take the region information for visualization
    regions = df[region_cols].values

    return predictions, regions

def read_mat_feats(feat_file,mat_file):

    regions = mat_to_np(mat_file,'boxes')
    predictions = mat_to_np(feat_file,'feats')

    # Reshape to match csv case...
    predictions = np.reshape(predictions,(np.shape(predictions)[1],np.shape(predictions)[0]))

    return predictions, regions

if __name__ == '__main__':

    query_mode = True

    # Test the functions

    if not query_mode:

        shot = 'shot26_6'
        frame = '00:00:27.56_000017'

        if params['net'] =='spp':

            # In case of SPP, we need to load object proposals separately

            mat_file = os.path.join(MAT_PATH,shot,frame+'.jpg.mat')
            predictions_file = os.path.join(FEAT_PATH,shot,frame+'.jpg.mat')

            predictions, regions = read_mat_feats(predictions_file,mat_file)

        else: # rcnn

            predictions , regions = read_csv_feats(os.path.join(FEAT_PATH,shot,frame+'.csv'))


        print "Predictions:" ,np.shape(predictions)
        print "Regions:" ,np.shape(regions)

    else:


        if params['net'] == 'spp':

            predictions = []

            for matfile in os.listdir(QUERY_FEAT_PATH):

                predictions_file = os.path.join(QUERY_FEAT_PATH,matfile)
                predictions.append(mat_to_np(predictions_file,'feats'))

            predictions = np.squeeze(np.array(predictions))
        else: # rcnn

            predictions , regions = read_csv_feats(QUERY_FEAT_PATH + '.csv')

        print "Predictions", np.shape(predictions)
