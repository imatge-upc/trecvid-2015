alpha = 0.5
f = open(os.path.join(BASELINE_RANKINGS,str(query) + '.rank'))

ranking_bow = pickle.load(f)[0:1000]
weights = pickle.load(f)[0:1000]
f.close()
weights = np.array(weights)


weights = (weights - np.min(weights) ) /( np.max(weights) - np.min(weights) )

f = open(os.path.join(SAVE_RCNN_RANKINGS,str(query) + '.rank'))

ranking_rcnn = pickle.load(f)
frames = pickle.load(f)
regions = pickle.load(f)
distances = pickle.load(f)
original_distance = pickle.load(f)

# From 0 to 1
original_distance = (original_distance - np.min(original_distance) ) / (np.max(original_distance) - np.min(original_distance))


new_scoring = (alpha*original_distance + (1-alpha)*weights)/2

new_ranking = np.array(ranking_bow)[np.argsort(new_scoring)[::-1]]

labels, num_relevant = eval.relnotrel(GROUND_TRUTH_FILE, str(query), new_ranking)
ap = eval.AveragePrecision(np.squeeze(labels),num_relevant)

print num_relevant, sum(sum(labels))

print "Baseline Average precision for query ", query, ':', ap

print "Baseline -in-subset- Average precision for query", query, ':', ap*num_relevant/sum(sum(labels))