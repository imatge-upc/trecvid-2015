addpath('../scripts/matlab/')

params = get_params()

for i=1:params.num_frames
    save_feats(i)
    display(i)
end
  
  