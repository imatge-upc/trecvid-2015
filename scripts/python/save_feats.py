import numpy as np
import pandas as pd
import os
import argparse
import time
import pickle
from rcnn import Detector
from get_params import get_params
import sys

params = get_params()

NUM_OUTPUT = 4096
CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

MODEL = params['caffe_path'] + '/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
DEPLOY = params['caffe_path'] +'/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'

def run_rcnn(script_name,input_csv,output_csv):

    if params['gpu']:
        arguments = [script_name, '--crop_mode=list', '--pretrained_model=' + MODEL, '--model_def=' + DEPLOY, '--gpu',  input_csv, output_csv]
    else:
        arguments = [script_name, '--crop_mode=list', '--pretrained_model=' + MODEL, '--model_def=' + DEPLOY,  input_csv, output_csv]

    sys.argv = arguments

    main(sys.argv)


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "input_file",
        help="Input txt/csv filename. If .txt, must be list of filenames.\
        If .csv, must be comma-separated file with header\
        'filename, xmin, ymin, xmax, ymax'"
    )
    parser.add_argument(
        "output_file",
        help="Output h5/csv filename. Format depends on extension."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                params['caffe_path'] +'models/bvlc_reference_caffenet/deploy.prototxt.prototxt'),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                params['caffe_path'] +'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="selective_search",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             params['caffe_path'] + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make detector.
    detector = Detector(args.model_def, args.pretrained_model,
            gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)

    if args.gpu:
        print 'GPU mode'

    # Load input.
    t = time.time()
    print('Loading input...')
    if args.input_file.lower().endswith('txt'):
        with open(args.input_file) as f:
            inputs = [_.strip() for _ in f.readlines()]
    elif args.input_file.lower().endswith('csv'):
        inputs = pd.read_csv(args.input_file, sep=',', dtype={'filename': str})
        
        inputs.set_index('filename', inplace=True)
    else:
        raise Exception("Unknown input file type: not in txt or csv.")

    print "csv loaded. Extracting features..."
    # Detect.
    if args.crop_mode == 'list':
        # Unpack sequence of (image filename, windows).
        images_windows = [
            (ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
            for ix in inputs.index.unique()
        ]
        detections = detector.detect_windows(images_windows)

        print "Features extracted. Storing..."
    else:
        images_windows = detector.detect_selective_search(inputs)
    
    print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                     time.time() - t))

    # Collect into dataframe with labeled fields.
    df = pd.DataFrame(detections)
    df.set_index('filename', inplace=True)
    df[COORD_COLS] = pd.DataFrame(
        data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)
    del(df['window'])

    # Save results.
    t = time.time()
    if args.output_file.lower().endswith('csv'):
    
        
        # csv
        # Enumerate the class probabilities.
        class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
        df[class_cols] = pd.DataFrame(data=np.vstack(df['prediction']),index=df.index, columns=class_cols)
        df.to_csv(args.output_file, cols=COORD_COLS + class_cols)
    else:
        # h5
        df.to_hdf(args.output_file, 'df', mode='w')
    print("Saved to {} in {:.3f} s.".format(args.output_file,
                                            time.time() - t))

def run(script_name,i):

    if params['database'] == 'db' or params['database'] == 'full':

        IMAGES = os.path.join(params['root'], '1_images', params['database'])
        DESCRIPTORS_PATH = os.path.join(params['root'], '5_descriptors',params['net'],params['database'] + params['year'])
        SELECTIVE_SEARCH_PATH = os.path.join(params['root'], '4_object_proposals', params['region_detector'], 'csv', params['database'] + params['year'])
        IMAGE_LIST = os.path.join(params['root'], '3_framelists', params['database'] + params['year'], params['query_name'] + '.txt')

    elif params['database'] == 'query':

        IMAGES = os.path.join(params['root'], '1_images', params['database'] + params['year'])
        DESCRIPTORS_PATH = os.path.join(params['root'], '5_descriptors',params['net'],params['database'] + params['year'])
        SELECTIVE_SEARCH_PATH = os.path.join(params['root'], '4_object_proposals', params['database'] + params['year'] + '_gt', 'csv')
        IMAGE_LIST = os.path.join(params['root'], '3_framelists', params['database'] + params['year'], params['query_name'] + '.txt')

    else: # query_selective_search

        IMAGES = os.path.join(params['root'], '1_images', 'query' + params['year'])
        DESCRIPTORS_PATH = os.path.join(params['root'], '5_descriptors',params['net'],params['database'] + params['year'])
        SELECTIVE_SEARCH_PATH = os.path.join(params['root'], '4_object_proposals', params['region_detector'], 'csv', params['database'] + params['year'])
        IMAGE_LIST = os.path.join(params['root'], '3_framelists', 'query' + params['year'], params['query_name'] + '.txt')


    if params['database'] =='db' or params['database'] == 'full':


        with(open(IMAGE_LIST,'r')) as f:
            image_list = f.readlines()

        image_path = image_list[i]

        image_name_parts = image_path.split('/')

        shot_name = image_name_parts[len(image_name_parts)-2]
        image_name = image_name_parts[len(image_name_parts)-1].rstrip()

        print image_name

        input_csv = SELECTIVE_SEARCH_PATH + '/' + shot_name + '/' +image_name[:-4] + '.csv'
        output_csv = DESCRIPTORS_PATH + '/' + shot_name + '/' + image_name[:-4] + '.csv'

        if not os.path.exists(DESCRIPTORS_PATH):
            os.makedirs(DESCRIPTORS_PATH)

        if not os.path.exists(DESCRIPTORS_PATH + '/' + shot_name):
            os.makedirs(DESCRIPTORS_PATH + '/' + shot_name)

        if not os.path.isfile(output_csv):
            print "File name is", input_csv
            run_rcnn(script_name,input_csv,output_csv)

        else:
            print "Skipped ", output_csv, '. File already existed.'



    elif params['database'] == 'query':
        
        if not os.path.exists(DESCRIPTORS_PATH):
            os.makedirs(DESCRIPTORS_PATH)
            
        input_csv = SELECTIVE_SEARCH_PATH + '/' + params['query_name'] + '.csv'
        output_csv = DESCRIPTORS_PATH + '/' + params['query_name'] + '.csv'

        if not os.path.isfile(output_csv):

            run_rcnn(script_name,input_csv,output_csv)
        else:
            print "Skipped ", output_csv, '. File already existed.'

    else:

        # Selective search descriptors for query
        if not os.path.exists(DESCRIPTORS_PATH):
            os.makedirs(DESCRIPTORS_PATH)

        for f in os.listdir(SELECTIVE_SEARCH_PATH):

            if params['query_name'] in f:

                input_csv = os.path.join(SELECTIVE_SEARCH_PATH,f)
                output_csv = os.path.join(DESCRIPTORS_PATH,f)

                if not os.path.isfile(output_csv):
                    run_rcnn(script_name,input_csv,output_csv)
                else:
                    print "Skipped ", output_csv, '. File already existed.'



if __name__ == "__main__":

    script_name = sys.argv[0]
    i = int(float(sys.argv[1]))

    run(script_name,i)







