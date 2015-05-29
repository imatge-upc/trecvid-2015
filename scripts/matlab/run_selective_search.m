function run_selective_search(id)

params = get_params();

if strcmp(params.dataset,'full')
    
    for i=id:id + params.batch_size - 1
        
        if i <= params.num_frames
            selective_search(params,i)
            display(i)
        end
    end
    
else
    
    
    for query=9099:9128
        
        query
        params.queryname = num2str(query);
        
        for i=2:5
            selective_search(params,i)
        end
        
    end
    
end
