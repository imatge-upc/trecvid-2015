# TRECVID Instance Search 2015

Tools for running the TRECVID 2015 Instance Search task at the Technical University of Catalonia.

## TODO:

- Automatically create data directories (preventing 'path does not exist' errors, add symbolic links to data and such)
- General cleanup
- Provide fine tuned models
- Provide detailed fine tuning steps and modifications


## Prerequisites:

- [Caffe](http://caffe.berkeleyvision.org/installation.html)

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn)

- [Selective search](https://github.com/sergeyk/selective_search_ijcv_with_python)


There is also code to use other networks:

- RCNN - (no installation needed - this is included already in Caffe's latest version)

- [ SPP_net](https://github.com/ShaoqingRen/SPP_net)


## Instructions:

The code is ready to reproduce the experiments using data from TRECVID Instance Search. The basic pipeline to follow would be:

0. The starting point is a list of ranked shots for each query
1. Generate frame lists from the shot lists with `scripts/python/create_image_list.py`
2. Compute selective search regions for database images with `scripts/matlab/selective_search.m`. This will run selective search for N images, so you can easily run this in multiple cores. You can also skip this step and use arbitrary locations at different scales by setting `params['use_proposals']` to `False`
3. Feature extraction with Fast-RCNN with `scripts/python/fast_rcnn_comp.py`. This script stores distances or scores instead of descriptors for all frames in the image lists
4. Merge distances to form ranking with `scripts/python/merge.py`
5. Evaluate ranking with `scripts/python/evaluate.py`
6. Display ranking with`scripts/python/display.py`

You should run steps 3 and 5 with `job_arrays`, which allow you to run N processes in parallel in the GPI computational service. 

In particular, for selective search you should run:

`sbatch --array=1-N:100 job_array/selective_search.sh` 

which would compute selective search regions for N images, in groups of 100. This '100' is specified in the matlab parameter `params.batch_size = 100` which you can change in `scripts/matlab/get_params.m`

And to run step 5 for all queries you should do:

`sbatch --array=9099-9128 job_array/merge.sh`

which will generate rankings for queries from 9099 to 9128.

Parameters are specified in `scripts/python/get_params.py` and `scripts/matlab/get_params.m` so that all scripts can work.

## Fine tuning:

In `scripts/python/prep_data_finetuning.py` I adapt query data to the Fast-RCNN format. Once that's done, you should adapt the Fast-RCNN code for fine tuning in order to read your data and ground truth.
[Here] (https://github.com/EdisonResearch/fast-rcnn/tree/master/help/train) is how to do it.

## Fine tuned models:

(To be uploaded)

TRECVID 2014
TRECVID 2015
