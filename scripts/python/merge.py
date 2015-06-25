from get_params import get_params
import pickle
import time
import sys
import fast_rcnn_comp

params = get_params()
params['query_name'] = str(sys.argv[1])

ts = time.time()
if params['query_name'] not in (9100,9113,9117):
    fast_rcnn_comp.merge_distances(params)
    print "Merged for query", params['query_name'], 'in', time.time() - ts, 'seconds.'