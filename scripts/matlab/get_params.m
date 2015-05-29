function params = get_params()

  params = struct();
  params.root = '/imatge/asalvador/work/trecvid/ins15/';
  
  params.gpu = true;
  params.queryname = '9099';
  params.lengthranking = 1000;
  params.regiondetector = 'selective_search';
  params.fastmode = false;
  params.batch_mode = true;
  params.net = 'spp';
  params.spp_path = '/imatge/asalvador/workspace/SPP_net/';
  params.year = '2014';
  params.dataset = 'query' ;
  params.num_frames = 63907;
  params.num_candidates = 20000;
  params.batch_size = 100;
