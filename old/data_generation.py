import os
import numpy as np
import glob
import pickle

def store_ground_truth(path_to_file,save_file):

    # Convert ground truth into dictionary for speed

    with open(path_to_file) as f:
        content = f.readlines()

    query = []
    shot = []
    relevance = []
    for line in content:

        chunk = line.split(' ')

        query.append(chunk[0])
        shot.append(chunk[2])
        relevance.append(chunk[3])

    print "Read the GT file."
    print "Creating GT dictionary..."

    # Create dictionary with ground truth

    ground_truth_dict = {}

    queries = range(9099,9129)
    for q in queries:

        print q

        query_positions = np.array( (np.array(query)==str(q) ),dtype=bool)
        positive_positions = np.array( (np.array(relevance) == '1\n'),dtype=bool)

        array_of_positives = np.array(shot)[np.logical_and(list(query_positions),list(positive_positions)) ]

        print "Number of positives:", np.shape(array_of_positives)

        ground_truth_dict[str(q)] = array_of_positives


    print "Done. Stored."

    pickle.dump(ground_truth_dict,open(save_file,"wb"))
    return []


def load_ranking(query,path_to_rankings):

    ranking = pickle.load(open( os.path.join(path_to_rankings,str(query)) + '.rank', 'rb') )

    return ranking

def frames_in_shot(shot_name,path_to_frames):

    frame_list = glob.glob( os.path.join(path_to_frames,shot_name) + '/*.jpg')

    return frame_list

def create_image_list(path_to_rankings,path_to_frames, save_path):

    queries = range(9069,9099)

    for query in queries:

        # Don't do this if the file is already there
        if not os.path.isfile(save_path + str(query) + '.p'):

            image_list = []

            # Load its ranking list
            ranking = load_ranking( str(query), path_to_rankings)
            ranking = ranking[0:100]

            # For each shot in the ranking...
            for shot in ranking:

                # Load the images in the shot
                images_in_shot = frames_in_shot(shot, path_to_frames)
                '''
                # Take only one frame every 4

                positions = np.arange(0,len(images_in_shot),4)
                images_in_shot = list(np.array(images_in_shot)[np.array(positions)])
                '''

                if len(image_list)==0:
                    image_list = images_in_shot
                else:
                    image_list = np.hstack((image_list,images_in_shot))

            pickle.dump( image_list, open(save_path + str(query) + '.p','wb') )
            print "Saved top 100 images for query", query

        else:
            print "Ranking already available for query", query

def create_image_list_2014(path_to_rankings,path_to_frames, save_path):

    queries = [9099]

    for query in queries:

        # Don't do this if the file is already there
        if not os.path.isfile(save_path + str(query) + '.p'):

            image_list = []

            # Load its ranking list
            ranking = load_ranking( str(query), path_to_rankings)
            ranking = ranking[0:1000]

            # For each shot in the ranking...
            for shot in ranking:

                # Load the images in the shot
                images_in_shot = frames_in_shot(shot, path_to_frames)
                '''
                # Take only one frame every 4

                positions = np.arange(0,len(images_in_shot),4)
                images_in_shot = list(np.array(images_in_shot)[np.array(positions)])
                '''

                if len(image_list)==0:
                    image_list = images_in_shot
                else:
                    image_list = np.hstack((image_list,images_in_shot))

            pickle.dump( image_list, open(save_path + str(query) + '.p','wb') )
            print "Saved top 100 images for query", query

        else:
            print "Ranking already available for query", query

if __name__ == "__main__":

    trecvid_path = '/imatge/asalvador/work/trecvid/'
    path_to_rankings = trecvid_path + 'rankings2014_nii/'
    path_to_frames = trecvid_path + 'images_1000_4_nii/'

    ground_truth_file = trecvid_path + 'ground_truth_files/ins.search.qrels.tv14'
    ground_truth_dict = trecvid_path + 'ground_truth_files/ins.search.qrels2014_dict.p'

    save_path = trecvid_path + 'image_lists_2014_nii/'

    print "Creating file with images..."

    create_image_list_2014(path_to_rankings , path_to_frames, save_path )

    print "Done."

    print "Storing ground truth in python-friendly dictionary..."

    store_ground_truth(ground_truth_file,ground_truth_dict)
    print "Done."
    print "All done."



