# TRECVID Instance Search 2015

Tools for running the TRECVID 2015 Instance Search task at the Technical University of Catalonia.
(Under construction)

### TODO:

- Automatically create data directories (preventing 'path does not exist' errors, add symbolic links to data and such)
- General cleanup
- Provide fine tuned models
- Provide detailed fine tuning steps and modifications


### Prerequisites:

- [Caffe](http://caffe.berkeleyvision.org/installation.html)

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn)

- [Selective search](https://github.com/sergeyk/selective_search_ijcv_with_python)


There is also code to use other networks:

- RCNN - (no installation needed - this is included already in Caffe's latest version)

- [ SPP_net](https://github.com/ShaoqingRen/SPP_net)

### Fine tuning:

Assuming query images are available, the first thing to do is to fine tune our model. To do this, you should first run `scripts/python/query_bboxes.py`, which will read the query ground truth masks and store the bounding box coordinates in csv files. 

Then, you should run `scripts/python/prep_data_finetuning.py`, which generates the necessary files in order to adapt query data to the Fast-RCNN format for fine tuning. You will see that this asumes that all query images are stored in the same path all together. Make sure that is true. 

Finally, you need to save the bounding box locations that will be used as candidates for each query. I do that in `scripts/matlab/save_boxes.m`.

Once that's done, we need to make this data readable for Fast R-CNN. I have included the changes that I have made to fast-rcnn code in order to make this happen. You can find them in the `fast-rcnn` folder. Notice that this is not the full fast-rcnn code. Only those paths containing scripts that I have created or modified are included. More specifically:

1. I created a new class for my dataset.

  In `fast-rcnn/lib/datasets/trecvid`. 

  This will read the data that we prepared before (images, ground truth and boxes) and store it in a Fast R-CNN friendly format. 

2. Modified `fast-rcnn/lib/datasets/factory.py` to run for our new class.

3. Then, I adapted `fast-rcnn/datasets/__init__.py` to use the new class, and I modified some parameters in `fast-rcnn/datasets/config.py`.

4. I slightly modified the solver, to create `models/VGG_CNN_M_1024/solver_trecvid.prototxt`.

5. I changed the last layers in the architecture and changed the number of outputs of the original network to fit my problem (30 classes). 

  The new network definition can be found in: `models/VGG_CNN_M_1024/train_trecvid.prototxt`.

So, In theory, at this stage you should only:

1. Run `fast-rcnn/lib/datasets/factory.py`. 

  This will prepare our data for fine tuning.

7. Train the network, running from the fast-rcnn root: 

  `./tools/train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/solver_trecvid.prototxt --weights data/fast_rcnn_models/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel --imdb trecvid_train`


When the process finishes, you will find the final model (and other snapshots of it) in `fast-rcnn-root/output/default/trecvid`.

[Here] (https://github.com/EdisonResearch/fast-rcnn/tree/master/help/train) you can find more information on how to adapt Fast R-CNN to your data.


### Instructions:

At this point, we are ready to use our model in the dataset from TRECVID Instance Search. The starting point is a ranking based on BoW, which we want to rerank using our Fast R-CNN model trained for TRECVID queries. The basic pipeline to follow would be:

1. Generate frame lists from the shot lists with `scripts/python/create_image_list.py`

2. [optional] Compute selective search regions for database images with `scripts/matlab/selective_search.m`. 

  This will run selective search for N images, so you can easily run this in multiple cores. 

  **Note 1:** You can also skip this step and use arbitrary locations at different scales by setting `params['use_proposals']` to `False`

3. Feature extraction with Fast-RCNN with `scripts/python/fast_rcnn_comp.py`. 

  This script stores distances or scores instead of descriptors for all frames in the image lists.

4. Merge distances to form ranking with `scripts/python/merge.py`

5. Evaluate ranking with `scripts/python/evaluate.py`

6. Display ranking with`scripts/python/display.py`


You should run steps 2 and 4 `job_arrays`, which allow to run N processes in parallel in the GPI computational service. 

In particular, for selective search you should run:

`sbatch --array=1-N:100 job_array/selective_search.sh` 

which would compute selective search regions for N images, in groups of 100. This '100' is specified in the matlab parameter `params.batch_size = 100` which you can change in `scripts/matlab/get_params.m`

And to run step 4 for all queries you should do:

`sbatch --array=9099-9128 job_array/merge.sh`

which will generate rankings for queries from 9099 to 9128.

Parameters are specified in `scripts/python/get_params.py` and `scripts/matlab/get_params.m` so that all scripts can work.

### Fine tuned models:

(To be uploaded)

TRECVID 2014


TRECVID 2015
