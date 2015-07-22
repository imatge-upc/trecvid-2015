from get_params import get_params
from create_image_list import frames_in_shot
import sys
import os
import numpy as np
params = get_params() # check get_params.py in the same directory to see the parameters
sys.path.insert(0,os.path.join(params['caffe_path'],'python'))
import caffe
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from evaluate import relnotrel, AveragePrecision
import time

layer = 'conv5'

def init_net(params):

    mean_file = os.path.join(params['places_model'], 'hybridCNN_mean.binaryproto')
    deploy_file =  os.path.join(params['places_model'], 'hybridCNN_deploy.prototxt')
    model_file =  os.path.join(params['places_model'], 'hybridCNN_iter_700000.caffemodel')
    
    # Transform mean file to numpy array
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean_file , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    
    out = np.resize(out,(3,227,227))
    
    caffe.set_mode_gpu()
    net = caffe.Classifier(deploy_file, model_file, mean=out, channel_swap=(2,1,0),raw_scale=255)
    
    return net

def pool_feats(feats):
    
    # Pool features from conv5
    size = 3
    scores = []
    for i in range(np.shape(feats)[1]):
        for ii in np.arange(1,13,size):
            for jj in np.arange(1,13,size):
    
                scores.append(np.max(feats[0,i,ii:ii+size,jj:jj+size]))
    
    scores = np.array(scores)
    
    return scores
    
def get_query_feats(params):
    
    query_feats = []
    for i in np.arange(4)+1:
            
        imagepath = os.path.join(params['root'],'1_images', 'query' + params['year'],params['query_name'],params['query_name']+'.'+ str(i)+'.src.bmp')
                
        scores = net.predict([caffe.io.load_image(imagepath)],oversample=False)
        
        
        scores = net.blobs[layer].data
       

        if 'conv' in layer:
            scores = pool_feats(scores)
        
        if len(query_feats) == 0:
            query_feats = scores
                   
        else:
            query_feats = np.vstack((query_feats,scores))
            
    return query_feats

def get_baseline_ranking(params):
    
    BASELINE_RANKING = os.path.join(params['root'], '2_baseline',params['baseline'],params['query_name'] + '.rank')
    shot_list = pickle.load(open( BASELINE_RANKING, 'rb') )
    
    if not params['database'] == 'gt_imgs':
        shot_list = shot_list[0:params['length_ranking']]
    
    return shot_list

def get_shot_feats(frames):
    
    shot_feats = []
    for frame in frames:
        scores = net.predict([caffe.io.load_image(frame)],oversample=False)
        scores = net.blobs[layer].data 
        if 'conv' in layer:
            scores = pool_feats(scores)
        
        if len(query_feats) == 0:
            shot_feats = scores
                   
        else:
            shot_feats = np.vstack((query_feats,scores))
            
    return shot_feats

def get_distance(query_feats,shot_feats):
    distances = pairwise_distances(shot_feats,query_feats,'euclidean')
    
    return distances
if __name__ == '__main__':
    net = init_net(params)
    
    print "Ranking for", layer, params['places_model']
    print '========'
    queries = range(9099,9129)
    
    for query in queries:
        if query not in (9100,9113,9117):
            
            params['query_name'] = str(query)
            
            query_feats = get_query_feats(params)
            
            shot_list = get_baseline_ranking(params)
            shot_scores = []
            
            i = 0
            for shot in shot_list:
                ts = time.time()
                frames = frames_in_shot(shot,params['root'] + '1_images/' + params['database'])
                
                shot_feats = get_shot_feats(frames)
                
                shot_distances = get_distance(shot_feats,query_feats)
                
                shot_distance = np.mean(shot_distances)
                
                shot_scores.append(shot_distance)
                #print "Processed shot", i, "in", time.time() - ts
                i = i + 1
                
            ranking = np.array(shot_list)[np.argsort(shot_scores)]
            
            # Evaluate
            GROUND_TRUTH_FILE = os.path.join(params['root'],'8_groundtruth','src','ins.search.qrels.tv14')

            labels, num_relevant = relnotrel(GROUND_TRUTH_FILE, params['query_name'], ranking)

            ap = AveragePrecision(np.squeeze(labels),num_relevant)
            
            print ap
        
            
            
