# TRECVID Instance Search 2015

Tools for running the TRECVID 2015 Instance Search task at the Technical University of Catalonia.
(Under construction)

## TODO:

- Automatically create data directories (preventing 'path does not exist' errors, add symbolic links to data and such)
- General cleanup
- Provide fine tuned models


## Prerequisites:

-[Caffe](http://caffe.berkeleyvision.org/installation.html)

At least one of the following*:

-RCNN - (no installation needed - this is included already in Caffe's latest version)

-[SPP_net](https://github.com/ShaoqingRen/SPP_net)

-[Fast-RCNN](https://github.com/rbgirshick/fast-rcnn)

*My latest experiments use Fast-RCNN, but there is code to compute RCNN and SPP features as well. Just note that at some point I stopped using them, so it might not be the greatest.

Python libraries:
-Pandas, numpy, scipy, sklearn

-[Selective search](https://github.com/sergeyk/selective_search_ijcv_with_python)

## Instructions:

The code is ready to reproduce the experiments using data from TRECVID Instance Search. The basic pipeline to follow would be:

1. Generate frame lists for your query and database images (`scripts/python/create_image_list.py`).
2. Compute selective search regions for database images (`scripts/matlab/selective_search.m`).
3. If needed, convert the selective search files to .csv to work with R-CNN code in python (`scripts/python/mat_to_csv.py`)
4. Feature extraction (`scripts/python/save_feats.py` for rcnn, `scripts/matlab/save_feats.m` for spp or `scripts/python/fast_rcnn_comp.py` for fastrcnn).
5. Compute distances from database descriptors to query descriptors. (`scripts/python/compute_distances.py`) - the code for fast-rcnn integrates this step already. If you use it, you can go directly to 6.
6. Merge distances to form ranking. (`scripts/python/rank.py`) 
7. Evaluate ranking. (`scripts/python/evaluate.py`)
8. Display ranking. (`scripts/python/display.py`)

Under `'test'` you can also find the scripts that I used to run some of the steps above for the whole dataset. 
In `'job_arrays'` you can find their bash equivalents (to use with sbatch in the GPI computational service.) This is mainly used to generate selective search proposals.

There are other scripts to collect data for SVM training as well (`scripts/python/select_samples.py`, which chooses which window samples over query images and computes the descriptors, and `scripts/python/train_svm.py` - which does the rest). 
This will train separate SVM models for each one of the queries, which can later be used to score database descriptors instead of using euclidean distances.
`scripts/python/fast_rcnn_comp.py` is already prepared to use SVMs if specified in the parameters.

Finally, I also fine tuned Fast-RCNN with TRECVid data. In `scripts/python/prep_data_finetuning.py` I adapt query data to the Fast-RCNN format. 
Here you can find my Fast-RCNN fork, with the necessary changes in the code to run on TRECVid data if you want to reproduce my experiments, or you can simply download the fine tuned models.

All the parameters are specified in `scripts/python/get_params.py` and `scripts/matlab/get_params.m` so that all scripts can work, so if you want to make changes you definitely need to make changes there.



