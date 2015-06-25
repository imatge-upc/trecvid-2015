from get_params import get_params
import numpy as np
import os
import pickle
import evaluate as eval
params = get_params()

def fuse(params):

    BOW_RANKINGS =  os.path.join(params['root'],'2_baseline',params['baseline'])
    FRCNN_RANKINGS = os.path.join(params['root'],'7_rankings',params['net'],params['database'] + params['year'], params['distance_type'])
    DPM_RANKINGS = os.path.join(params['root'],'2_baseline','nii_dpm')
    GROUND_TRUTH_FILE = os.path.join(params['root'],'8_groundtruth','src','ins.search.qrels.tv14')
    alpha = params['fusion_alpha']
    f = open(os.path.join(BOW_RANKINGS,str(query) + '.rank'))



    # Load BoW
    ranking_bow = pickle.load(f)[0:params['length_ranking']]
    weights_bow = pickle.load(f)[0:params['length_ranking']]
    f.close()
    weights_bow = np.array(weights_bow)
    # Normalize weights
    weights_bow = (weights_bow - np.min(weights_bow) ) /( np.max(weights_bow) - np.min(weights_bow) )

    # Load DPM
    f = open(os.path.join(DPM_RANKINGS,str(query) + '.rank'))

    ranking_dpm = pickle.load(f)
    weights_dpm = pickle.load(f)

    f.close()

    # Delete unwanted
    weights_dpm = dpm1000(ranking_dpm,ranking_bow,weights_dpm)
    weights_dpm = np.array(weights_dpm)
    # Normalize weights
    weights_dpm = (weights_dpm - np.min(weights_dpm) ) /( np.max(weights_dpm) - np.min(weights_dpm) )


    # Load FRCNN

    f = open(os.path.join(FRCNN_RANKINGS,str(query) + '.rank'))
    ranking_frcnn = pickle.load(f)
    frames = pickle.load(f)
    regions = pickle.load(f)
    distances = pickle.load(f)
    weights_frcnn = pickle.load(f)
    
    # Normalize
    weights_frcnn = (weights_frcnn - np.min(weights_frcnn) ) / (np.max(weights_frcnn) - np.min(weights_frcnn))


    if params['fusion-scheme'] == 'bow-dpm':
        new_scoring = (alpha*weights_dpm + (1-alpha)*weights_bow)
    elif params['fusion-scheme'] == 'bow-frcnn':
        new_scoring = (alpha*weights_frcnn + (1-alpha)*weights_bow)
    elif params['fusion-scheme'] == 'frcnn-dpm':
        new_scoring = (alpha*weights_frcnn + (1-alpha)*weights_dpm)
    else:
        new_scoring = (0.25*weights_frcnn + 0.5*weights_bow + 0.25*weights_dpm)

    new_ranking = np.array(ranking_bow)[np.argsort(new_scoring)[::-1]]
    
    labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), new_ranking)
    ap = eval.AveragePrecision(np.squeeze(labels),num_relevant)
    
    #print num_relevant, sum(sum(labels))
    
    print ap
    return ap

def dpm1000(dpm_rank,bow_rank,dpm_weights):

    new_dpm_weights = []

    for shot in bow_rank:
        idx = dpm_rank.index(shot)
        new_dpm_weights.append(dpm_weights[idx])

    return new_dpm_weights

if __name__ == "__main__":
    
    params = get_params()
    
    queries = range(9099,9129)
    ap = []
    for query in queries:
        if query not in (9100,9113,9117):
            params['query_name'] = str(query)
            ap.append(fuse(params))
    print np.mean(ap)