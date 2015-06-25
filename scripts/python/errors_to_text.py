import pickle
import numpy as np
errors = pickle.load(open('errors.p','rb'))


errors = np.unique(errors)


for e in errors:
    print e.rstrip()
