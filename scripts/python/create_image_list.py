from get_params import get_params
import os
import numpy as np
import glob
import pickle

""" Used to create baseline ranklists to sort using fast-rcnn. """
def load_ranking(query,path_to_rankings):

    ranking = pickle.load(open( os.path.join(path_to_rankings,str(query)) + '.rank', 'rb') )

    return ranking

def frames_in_shot(shot_name,path_to_frames):

    frame_list = glob.glob( os.path.join(path_to_frames,shot_name) + '/*.jpg')

    return frame_list

def create(params):

    query = params['query_name']

    save_path = params['root'] + '3_framelists/' + params['database'] + params['year']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    image_list = []
    if not os.path.isfile(save_path + query + '.txt'):

        txt_file = open(os.path.join(save_path,query + '.txt'),'w')

        # Load its ranking list
        ranking = load_ranking( query, params['root'] + '2_baseline/nii_bow/')
        ranking = ranking[0:params['length_ranking']]

        # For each shot in the ranking...
        for shot in ranking:
           
            # Load the images in the shot
            images_in_shot = frames_in_shot(shot, params['root'] + '1_images/' + params['database'])


            if len(image_list)==0:
                image_list = images_in_shot
            else:
                image_list = np.hstack((image_list,images_in_shot))

        txt_file.writelines(["%s\n" % item  for item in image_list])
        txt_file.close()

        print "Saved top images for query", query
        print "Number of images:", len(image_list)

    else:
        print "Ranking already available for query", query
    
    return image_list

def create_all(params):

    shot_path = params['root'] + '1_images/' + params['database']

    image_list = []
    for shot in os.listdir(shot_path):

        print shot
        images_in_shot = frames_in_shot(shot, shot_path)

        if len(image_list)==0:
            image_list = images_in_shot
        else:
            image_list = np.hstack((image_list,images_in_shot))

    return image_list
    

def shotlist(params):

    shot_path = params['root'] + '1_images/' + params['database']

    shot_list = []
    for shot in os.listdir(shot_path):

        shot_list.append(shot)
        

    return shot_list    

    

if __name__ == "__main__":
    
    
    params = get_params()
    
    
    queries = range(9099,9129)
    
    if params['rerank_bool']:
        
        image_list = []
        for query in queries:
            if query not in (9100,9113,9117):
                
                params['query_name'] = str(query)
                imlist = create(params)
                if len(image_list)==0:
                    image_list = imlist
                else:
                    image_list = np.hstack((image_list,imlist))
        save_path = params['root'] + '3_framelists/' + params['database'] + params['year']
        all_frames_file = open(os.path.join(save_path,'all_frames' + '.txt'),'w')
        all_frames_file.writelines(["%s\n" % item  for item in image_list])
        all_frames_file.close()
        print "Done. Total of ", len(image_list),' frames'
    
    else:
        shot_list = shotlist(params)
        image_list = create_all(params)
        
        save_path = params['root'] + '3_framelists/' + params['database'] + params['year'] + '_fullrank'
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        for query in queries:
            if query not in (9100,9113,9117):
                txt_file = open(os.path.join(save_path,str(query)+ '.txt'),'w')
                txt_file.writelines(["%s\n" % item  for item in image_list])
                txt_file.close()
        
        save_path = params['root'] + '2_baseline/' + params['database'] + params['year'] + '_fullrank'
    
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        for query in queries:
            if query not in (9100,9113,9117):
                txt_file = open(os.path.join(save_path,str(query) + '.txt'),'w')
                txt_file.writelines(["%s\n" % item  for item in shot_list])
                txt_file.close()