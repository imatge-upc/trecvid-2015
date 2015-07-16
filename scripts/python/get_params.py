def get_params():

    params = {'root': '/imatge/asalvador/work/trecvid/ins15/', # all data is stored here
              'gpu': True, # use gpu for feature extraction or not
              'query_name': 'all_frames',
              'height' : 768,
              'width' : 576,
              'region_detector': 'selective_search',
              'net': 'fast-rcnn', # rcnn and sppnet were used in the early stages
              'length_ranking': 1000, # length of the list to rerank
              'fusion_alpha':0.5,
              'use_proposals':False,
              'database': 'full',
              'fusion-scheme':'all', # 'bow-frcnn', 'frcnn-dpm', 'all'
              'year': '2014',
              'batch_size': 3000, # number of boxes that can be processed at once
              'baseline': 'nii_bow', #nii_bow
              'display_baseline': False,
              'delete_mode': False,
              'rerank_bool': False,
              'saliency_model': '/imatge/asalvador/work/caffe/models/salnet/salnet/model',
              'places_model': '/imatge/asalvador/work/caffe/models/places-hybridCNN',
              'distance_type':'scores01-sw6300-full',
              'caffe_path': '/usr/local/src/fast-rcnn/fast-rcnn/caffe-fast-rcnn', #where caffe is installed and compiled
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