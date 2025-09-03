## Installing necessary libraries

import pandas as pd
import sklearn

## Loading the dataset
beers = pd.read_csv('data/beer_reviews.csv')

beers = beers.drop_duplicates(subset=['beer_name'])  # Dropping duplicate beers by name to ensure data quality

beers = beers.dropna(subset=['beer_style'])  # Dropping rows where 'beer_style' is NaN

## Test statement displaying the first few rows of the dataset to verify it loaded correctly
## print(beers.head())

## The following statement creates a list of unique beer types in the dataset that we will use later.
beerTypes = beers['beer_style'].unique()

## This is a test statement used to provide insight for development purposes.
## It counts the number of unique beer types in the dataset and prints it out.
## n_types = len(beerTypes)
## print (f'There are {n_types} different types of beer in the dataset.')

## Creating an instance of 'oneHotEncoder' to convert categorical data into a format that can be provided to ML algorithms.
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_beer_types = encoder.fit_transform(beers[['beer_style']])

## We will be using a nearest neighbor algorithm to find similar beers based on their attributes.
from sklearn.neighbors import NearestNeighbors

recommender = NearestNeighbors(metric='euclidean')

recommender.fit(encoded_beer_types)  # This statement fits the nearest neighbor model to the encoded beer types.

## The following are test statements used to ensure the function of the recommender system. The final output will take these values from the user.

beer_index = 13

num_recs = 5

distances, indices = recommender.kneighbors(encoded_beer_types[beer_index].reshape(1, -1), n_neighbors=num_recs)

print(f"Recommendations for {beers.iloc[beer_index]['beer_name']} ({beers.iloc[beer_index]['beer_style']}):")
for idx in indices[0]:
    print(f"- {beers.iloc[idx]['beer_name']} ({beers.iloc[idx]['beer_style']}) from {beers.iloc[idx]['brewery_name']} \n")