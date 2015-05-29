import sys

sys.path.insert(0,'../scripts/python')
from get_params import get_params
import save_feats


params = get_params()

for i in range(params['num_frames']):

    save_feats.run('save_feats.py',i)
