import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, 'utils')
from preprocessing_utils import read_data_names, animate_img


path_to_external_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ/data/'

x,y = read_data_names(path_to_external_data)

all_characteristics = []
all_mean = []
for i in range(len(x)):
    x_data = np.load(os.path.join(path_to_external_data,x[i]))
    #y_data = np.load(os.path.join(path_to_raw_data,y[i]))
    _dist_characteristics = {
            "Name" : x[i][:-6],
            "Mean": np.mean(x_data),
            "Var": np.var(x_data),
            "Median" : np.median(x_data),
            "Min" : np.min(x_data),
            "Max" : np.max(x_data),
            "95 Percentile" : np.percentile(x_data, 95),
            "5 Percentile" : np.percentile(x_data, 5),
            "Max - Min" : np.max(x_data) - np.min(x_data)
        }

    all_characteristics.append(_dist_characteristics)
characteristic_names = all_characteristics[0].keys()

pd.DataFrame(all_characteristics).to_csv('/scratch/lhauptmann/segmentation_3D/data_analysis/data_characteristics.csv')