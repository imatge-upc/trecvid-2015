
function selective_search(params,i)


switch params.dataset
    case 'db'
        
    
        SAVE_PATH = [params.root  '4_object_proposals/'  params.regiondetector  '/mat/'   params.dataset  params.year];
        image_list = [params.root  '3_framelists/'  params.dataset  params.year  '/'  params.queryname  '.txt'];
        image_list = [params.root  '3_framelists/'  params.dataset  params.year  '/'  'errors.txt'];
    
    case 'full'
        
    
        SAVE_PATH = [params.root  '4_object_proposals/'  params.regiondetector  '/mat/'   params.dataset  params.year];
        image_list = [params.root  '3_framelists/'  params.dataset  params.year  '/'  params.queryname  '.txt'];
        %image_list = [params.root  '3_framelists/'  params.dataset  params.year  '/'  'errors.txt'];
        
    case 'gt_imgs'
        
    
        SAVE_PATH = [params.root  '4_object_proposals/'  params.regiondetector  '/mat/'   params.dataset];
        image_list = [params.root  '3_framelists/'  params.dataset  '/'  params.queryname  '.txt'];
        
    case 'query'
        
        SAVE_PATH = [params.root  '4_object_proposals/'  params.regiondetector  '/mat/'   params.dataset  params.year  '/'  params.queryname];
        image_list = [params.root  '4_object_proposals/'  params.dataset  params.year  '_gt/csv/'  params.queryname  '.csv'];
        
end

addpath(genpath(params.spp_path));

image_list = cellstr(image_list);
image_list = image_list{1};
fid = textread(image_list, '%s','delimiter', '\n');

imname = fid(i);
imname = imname{1}

switch params.dataset

    case 'db'
        if (exist(SAVE_PATH, 'dir') ~= 7)
            mkdir(SAVE_PATH)
        end
        im = imread(imname);
        shot = strsplit(imname,'/');
        shot = shot(length(strsplit(imname,'/')) - 1);
        frame = strsplit(imname,'/');
        frame = frame(length(strsplit(imname,'/')));
        
        
        shot_folder = fullfile(SAVE_PATH,shot);
        shot_folder = shot_folder{1};
        
        if (exist(shot_folder, 'dir') ~= 7)
            mkdir(shot_folder)
        end
        
        file_to_save = fullfile(shot_folder, strcat(frame{1},'.mat'));
                
    case 'full'
    
        if (exist(SAVE_PATH, 'dir') ~= 7)
            mkdir(SAVE_PATH)
        end
        im = imread(imname);
        shot = strsplit(imname,'/');
        shot = shot(length(strsplit(imname,'/')) - 1);
        frame = strsplit(imname,'/');
        frame = frame(length(strsplit(imname,'/')));
        
        
        shot_folder = fullfile(SAVE_PATH,shot);
        shot_folder = shot_folder{1};
        
        if (exist(shot_folder, 'dir') ~= 7)
            mkdir(shot_folder)
        end
        
        file_to_save = fullfile(shot_folder, strcat(frame{1},'.mat'));
        
    case 'gt_imgs'
    
        if (exist(SAVE_PATH, 'dir') ~= 7)
            mkdir(SAVE_PATH)
        end
        im = imread(imname);
        shot = strsplit(imname,'/');
        shot = shot(length(strsplit(imname,'/')) - 1);
        frame = strsplit(imname,'/');
        frame = frame(length(strsplit(imname,'/')));
        
        
        shot_folder = fullfile(SAVE_PATH,shot);
        shot_folder = shot_folder{1};
        
        if (exist(shot_folder, 'dir') ~= 7)
            mkdir(shot_folder)
        end
        
        file_to_save = fullfile(shot_folder, strcat(frame{1},'.mat'));
        
    case 'query'
    
        if (exist(SAVE_PATH, 'dir') ~= 7)
            mkdir(SAVE_PATH)
        end
        
        frame = strsplit(imname,',');
        frame = frame(1);
        frame = frame{1};
        
        im = imread(frame);
        
        frame_name = strsplit(frame,'/');
        frame_name = frame_name(length(frame_name));
        file_to_save = fullfile(SAVE_PATH,strcat(frame_name{1},'.mat'));
end


if (exist( file_to_save, 'file') ~=2) 
    th = tic();
    boxes = selective_search_boxes(im, params.fastmode);
    save(file_to_save, 'boxes')
    
    display('Saved!')
    fprintf('done (in %.3fs).\n', toc(th));
    length(boxes)
else
    
    display('File already existed ! ')
end