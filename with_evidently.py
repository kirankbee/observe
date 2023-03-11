# Created by kiran at 3/11/23
# Enter feature name here
# Enter feature description here
# Last modified by kiran at 3/11/23
# Deployment Scenario linux/iOS for deploying it on GKE must have us.gcr.io /
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, NumTargetDriftTab,CatTargetDriftTab

# fetch a reference data set
reference_data = \
pd.read_csv("training_data.csv", header=None,
            names=[ "day{}".format(i) for i in \
                    range(0,14) ]+["target"] )