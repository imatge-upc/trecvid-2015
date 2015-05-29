function save_feats(i)

params = get_params();

% Switch to know which images we are extracting features from, and where to
% load the data in each case
switch params.dataset
    case 'db'
        
        SAVE_PATH = fullfile(params.root ,'5_descriptors', params.net , strcat(params.dataset , params.year));
        REGION_PATH = fullfile(params.root, '4_object_proposals' , params.regiondetector , 'mat' ,  strcat(params.dataset, params.year));
        image_list = fullfile(params.root ,'3_framelists', strcat(params.dataset,params.year),  strcat(params.queryname, '.txt'));
    
    case 'full'
    
        SAVE_PATH = fullfile(params.root ,'5_descriptors', params.net , strcat(params.dataset , params.year));
        REGION_PATH = fullfile(params.root, '4_object_proposals' , params.regiondetector , 'mat' ,  strcat(params.dataset, params.year));
        image_list = fullfile(params.root ,'3_framelists', strcat(params.dataset,params.year),  strcat(params.queryname, '.txt'));
    
    case 'query_selective_search'
        
        SAVE_PATH = fullfile(params.root,'5_descriptors', params.net,strcat(params.dataset ,params.year));
        REGION_PATH = fullfile(params.root, '4_object_proposals', params.regiondetector, 'mat',  strcat(params.dataset,params.year), params.queryname);
        image_list = fullfile(params.root,'3_framelists',strcat(params.dataset,params.year) ,strcat( params.queryname, '.txt'));
        
        
    case 'query'
        SAVE_PATH = fullfile(params.root,'5_descriptors', params.net, + strcat('query', params.year), params.queryname);
        image_list = fullfile(params.root,'4_object_proposals',strcat(strcat('query', params.year), '_gt'),'csv', strcat( params.queryname ,'.csv'));
        
end

fid = textread(image_list, '%s','delimiter', '\n');

imname = fid(i);
im = imname{1};

if strcmp(params.dataset,'query')
    imname = fid(i+1);
    im = imname{1};
end

if strcmp(params.dataset,'db') || strcmp(params.dataset,'full')
    
    % In this case we are extracting features for the database. We need the name of the shot and the frame, and create the shot folder if it does not exist
    
    % Get shot and frame names
    shot = strsplit(im,'/');
    shot = shot(length(strsplit(im,'/')) - 1);
    frame = strsplit(im,'/');
    frame = frame(length(strsplit(im,'/')));
    
    % Prepare the directory name
    shot_folder = fullfile(SAVE_PATH,shot);
    shot_folder = shot_folder{1};
    
elseif strcmp(params.dataset,'query')
    
    % In this case we are extracting features for the query images using the ground truth bounding boxes. We can read everything from the image_list (boxes included!)
    
    % It means we have query images (no 'shot' or 'frame' info)
    shot_folder = SAVE_PATH;
    
    line_parts = strsplit(im,',');
    im = line_parts(1);
    im = im{1};
    display(im)
    
    frame = strsplit(im,'/');
    frame = frame( length( frame ) );
    display(frame)
    
    
    ymin = line_parts(2);
    xmin = line_parts(3);
    ymax = line_parts(4);
    xmax = line_parts(5);
    
    boxes = [ str2num( ymin{1} ) str2num( xmin{1} ) str2num( ymax{1} ) str2num( xmax{1} ) ]
else
    
    % This case is to extract features for query images, but using selective search proposals
    
    frame = strsplit(im,',');
    frame = frame( length( frame ) );
    shot_folder = SAVE_PATH;
    
end

% Create the shot folder if it does not exist
if (exist(shot_folder, 'dir') ~= 7)
    mkdir(shot_folder);
end


% Load selective search boxes in cases where needed
if strcmp(params.dataset,'db') || strcmp(params.dataset,'full')
    
    % This loads precomputed selective search 'boxes'
    load(fullfile(REGION_PATH,[shot{1}  '/'  frame{1}  '.mat']) );
    % compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
    boxes = boxes(:, [2 1 4 3]);
    boxes = boxes(1:min(size(boxes,1),params.num_candidates),:);
    
elseif strcmp(params.dataset,'query_selective_search')
    
    load(fullfile(REGION_PATH, strcat(frame{1},'.mat')) );
    % compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
    boxes = boxes(:, [2 1 4 3]);
    boxes = boxes(1:min(size(boxes,1),params.num_candidates),:);
    
end


% Compute CNN features for the image, if they have not been stored yet

display( fullfile(shot_folder,strcat(frame{1}, '.mat')) )

if (exist( fullfile(shot_folder,strcat(frame{1},'.mat')), 'file') ~=2)
    
    % Init SPP
    spp_model_file = fullfile(params.spp_path,'data/spp_model/VOC2007/spp_model.mat');
    
    if ~exist(spp_model_file, 'file')
        error('%s not exist ! \n', spp_model_file);
    end
    try
        load(spp_model_file);
    catch err
        fprintf('load spp_model_file : %s\n', err.message);
    end
    caffe_net_file     = fullfile(params.spp_path, 'data/cnn_model/Zeiler_conv5_new/Zeiler_conv5');
    caffe_net_def_file = fullfile(params.spp_path, 'data/cnn_model/Zeiler_conv5_new/Zeiler_spm_scale224_test_conv5.prototxt');
    
    if params.gpu
        clear mex;
        g = gpuDevice(1);
    end
    
    addpath(genpath(params.spp_path));
    
    caffe('init', caffe_net_def_file, caffe_net_file);
    caffe('set_phase_test');
    
    if params.gpu
        spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
        caffe('set_mode_gpu');
    else
        caffe('set_mode_cpu');
    end
    
    spm_im_size = [480 576 688 874 1200];
    % spm_im_size = [ 688 ];
    
    
    % Compute feats
    im
    feats = spp(imread(im), boxes, spp_model, spm_im_size, params.gpu);
    
    % Saving
    save(fullfile(shot_folder,strcat(frame{1}, '.mat')),'feats')
    
    display('Saved')
    
    caffe('release');
else
    display('File already existed. Skipping...')
end




