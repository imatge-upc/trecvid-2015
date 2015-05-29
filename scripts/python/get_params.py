def get_params():

    params = {'root': '/imatge/asalvador/work/trecvid/ins15/', # all data is stored here
              'gpu': True, # use gpu for feature extraction or not
              'query_name': '9113',
              'region_detector': 'selective_search',
              'net': 'fast-rcnn', # rcnn and sppnet were used in the early stages
              'length_ranking': 1000, # length of the list to rerank
              'database': 'full',
              'year': '2014',
              'batch_size': 3000, # number of boxes that can be processed at once
              'baseline': 'nii_bow',
              'display_baseline': False,
              'delete_mode': False,
              'distance_type':'scores03', # euclidean, or scores
              'caffe_path': '/imatge/asalvador/caffe/', #where caffe is installed and compiled
              'fast_rcnn_path': '/imatge/asalvador/workspace/fast-rcnn', # where fast-rcnn is installed and compiled
              'num_frames': 1614, # unused
              'num_candidates':2000, # max number of object candidates to use at test time
              'split_percentage':0.8, # split train/val for svm
              'additional_negatives': True,
              'num_additional': 1000000,
              'svm_iterations': 1,
              'min_negatives': 1,
              'layer' : 'cls_score_trecvid', #cls_prob
              'net_name': 'trecvid'}

    return params
