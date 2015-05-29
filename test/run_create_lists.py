import sys
import os
import numpy as np

sys.path.insert(0,'../scripts/python/')
import create_image_list

from get_params import get_params



image_list = []
params = get_params()

save_path = params['root'] + '3_framelists/' + params['database'] + params['year']
txt_file = open(os.path.join(save_path,'all_frames' + '.txt'),'w')


queries = range(9099,9129)

for query in queries:
    
    if query not in [9100,9113,9117]:
        params['query_name'] = str(query)
        images_in_query = create_image_list.create(params)
      
        if len(image_list)==0:
            image_list = images_in_query
        else:
            image_list = np.hstack((image_list,images_in_query))
        

txt_file.writelines(["%s\n" % item  for item in image_list])
txt_file.close()

print np.shape(image_list)
