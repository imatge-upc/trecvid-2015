params = get_params();

SAVE_PATH = [params.root  '4_object_proposals/'  params.regiondetector  '/mat/'   'query'  params.year  '/'];

image_db = '/imatge/asalvador/work/trecvid/ins15/1_images/trecvid/';

output_filename = [image_db '/selective_search/train_extensive.mat'];

image_filenames = textread([image_db 'imagesets/train.txt'], '%s', 'delimiter', '\n');

all_boxes = {};
for i=1:length(image_filenames)
    
    filename = image_filenames{i};
    query = strsplit(filename,'.');
    query = query{1}
    
    filename = [SAVE_PATH query '/' filename '.bmp.mat'];
    
    load(filename)
    
    all_boxes{i} = boxes;
    
end
    
    
    
save(output_filename, 'all_boxes', '-v7');