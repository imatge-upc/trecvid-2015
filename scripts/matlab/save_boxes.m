image_db = '/imatge/asalvador/work/trecvid/ins15/1_images/finetuning/2015/';
image_filenames = textread([image_db 'imagesets/train.txt'], '%s', 'delimiter', '\n');


sizes_x = [32,64,128,256,512];
sizes_y = [32,64,128,256,512];
all_boxes = {};

for iii = 1:length(image_filenames)
    boxes = [];
    num = 1;
    im = imread([image_db '/images/' image_filenames{iii} '.png']);
    
    [h,w,c] = size(im);
    
    for ii=1:length(sizes_x)
        for jj=1:length(sizes_y)
        
            sx = sizes_x(ii);
            sy = sizes_y(jj);
            
            stepx = sx/2;
            stepy = sy/2;


            for i=1:stepx:h
                for j=1:stepy:w

                    boxes(num,:) = [i, j, min(i+sx,h), min(j+sy,w)];
                    num = num + 1;

                end
            end
        end
        
        
    end
    
    size(boxes)
    
    all_boxes{iii} = boxes;
end
save([image_db '/selective_search/train_sw.mat'], 'all_boxes', '-v7');

    
