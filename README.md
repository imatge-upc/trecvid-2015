# TRECVID Instance Search 2015

Tools for running the TRECVID 2015 Instance Search task at the Technical University of Catalonia.
(Under construction)

## TODO:

- Automatically create data directories (preventing 'path does not exist' errors, add symbolic links to data and such)
- General cleanup
- Provide fine tuned models


## Prerequisites:

- [Caffe](http://caffe.berkeleyvision.org/installation.html)

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn)

- [Selective search](https://github.com/sergeyk/selective_search_ijcv_with_python)


There is also code to use other networks:

- RCNN - (no installation needed - this is included already in Caffe's latest version)

- [ SPP_net](https://github.com/ShaoqingRen/SPP_net)


## Instructions:

The code is ready to reproduce the experiments using data from TRECVID Instance Search. The basic pipeline to follow would be:

0. The starting point is a list of ranked shots for each query.
1. Generate frame lists from the shot lists (`scripts/python/create_image_list.py`).
2. Compute selective search regions for database images (`scripts/matlab/selective_search.m`). This will run selective search for N images, so you can run this in multiple cores.
3. Feature extraction with Fast-RCNN `scripts/python/fast_rcnn_comp.py`. This script already stores distances or scores instead of descriptors.
4. Merge distances to form ranking. (`scripts/python/rank.py`) 
5. Evaluate ranking. (`scripts/python/evaluate.py`)
6. Display ranking. (`scripts/python/display.py`)

Under `test` you can also find the scripts that I used to run some of the steps above for the whole dataset. 
In `job_arrays` you can find their bash equivalents (to use with sbatch in the GPI computational service.) This is mainly used to generate selective search proposals.

There are other scripts to collect data for SVM training as well (`scripts/python/select_samples.py`, which chooses which window samples over query images and computes the descriptors, and `scripts/python/train_svm.py` - which does the rest). 
This will train separate SVM models for each one of the queries, which can later be used to score database descriptors instead of using euclidean distances.
`scripts/python/fast_rcnn_comp.py` is already prepared to use SVMs if specified in the parameters.

All the parameters are specified in `scripts/python/get_params.py` and `scripts/matlab/get_params.m` so that all scripts can work.

## Fine tuning:

In `scripts/python/prep_data_finetuning.py` I adapt query data to the Fast-RCNN format. Once that's done, you should adapt the Fast-RCNN code for fine tuning in order to read your data and ground truth.
[Here] (https://github.com/EdisonResearch/fast-rcnn/tree/master/help/train) is how to do it.

## Fine tuned models:

(To be uploaded)

TRECVID 2014


TRECVID 2015
