function run_selective_search(id)

	params = get_params()

	for i=id:id + 20 - 1
		selective_search(i)
		display(i)
	end
