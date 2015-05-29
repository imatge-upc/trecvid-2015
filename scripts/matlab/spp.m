function feat = spp_detect(im, boxes, spp_model, spm_im_size, use_gpu)
% Adapted from spp code written by Ross Girshick
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if isempty(boxes)
    feat = {};
    return;
end

% extract features from candidates (one row per candidate box)
fprintf('Extracting CNN features from regions...');
th = tic();

% calc_fc_in_matlab = true;
feat = spp_features_convX(im, spm_im_size, [], use_gpu);
feat = spp_features_convX_to_poolX(spp_model.spp_pooler, feat, boxes, false);
feat = spp_poolX_to_fcX(feat, spp_model.training_opts.layer, spp_model, use_gpu);
feat = spp_scale_features(feat, spp_model.training_opts.feat_norm_mean);

fprintf('done (in %.3fs).\n', toc(th));
