## Installing necessary libraries

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

## Loading the dataset
beers = pd.read_csv('data/beer_reviews.csv')

## Test statement displaying the first few rows of the dataset to verify it loaded correctly
## print(beers.head())
