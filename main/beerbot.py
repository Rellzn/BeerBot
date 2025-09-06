## Installing necessary libraries

import pandas as pd
import sklearn
import tkinter as tk
from tkinter import *

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

## beer_index = 13

## num_recs = 5

## distances, indices = recommender.kneighbors(encoded_beer_types[beer_index].reshape(1, -1), n_neighbors=num_recs)

## print(f"Recommendations for {beers.iloc[beer_index]['beer_name']} ({beers.iloc[beer_index]['beer_style']}):")
## for idx in indices[0]:
##    print(f"- {beers.iloc[idx]['beer_name']} ({beers.iloc[idx]['beer_style']}) from {beers.iloc[idx]['brewery_name']} \n")

def get_beer_recommendations(beer_index): ## This function takes in a beer index and the number of recommendations to return a list of strings with beer information for each recommendation.
    distances, indices = recommender.kneighbors(encoded_beer_types[beer_index].reshape(1, -1), n_neighbors=5) ## UI will fetch these values from the user.
    recommendations = []
    for idx in indices[0]:
        recommendations.append(f"- {beers.iloc[idx]['beer_name']} ({beers.iloc[idx]['beer_style']}) from {beers.iloc[idx]['brewery_name']}")
    return recommendations


## Creating the GUI

## Main window
root = Tk()

## Window properties
root.title("BeerBot 3000")
root.geometry("600x400")

## Title label
title_label = Label(root, text="BeerBot 3000", font=("Helvetica", 24))
title_label.pack(pady=10)

def update_listbox(data, listbox):
    """Clears and updates the listbox with new data."""
    listbox.delete(0, END)
    for item in data:
        listbox.insert(END, item)

def check_input(event, entry_var, data_list, listbox):
    """Filters data_list based on entry content and updates listbox."""
    typed_text = entry_var.get().lower()
    if not typed_text:
        filtered_data = data_list
    else:
        filtered_data = [item for item in data_list if typed_text in item.lower()]
    update_listbox(filtered_data, listbox)

def fill_entry(event, entry_var, listbox):
    """Fills the entry with the selected listbox item."""
    if listbox.curselection():  # Check if an item is actually selected
        selected_item = listbox.get(ANCHOR)
        entry_var.set(selected_item)
        listbox.delete(0, END) # Optionally clear suggestions after selection

def create_autocomplete_search_bar(root, data_list):
    """Creates an autocompleting search bar."""
    entry_var = tk.StringVar()

    entry = tk.Entry(root, textvariable=entry_var, font=("Helvetica", 16))
    entry.pack(pady=10)

    listbox = tk.Listbox(root, width=40, height=8, font=("Helvetica", 14))
    listbox.pack(pady=5)

    # Initial population of the listbox
    update_listbox(data_list, listbox)

    # Bind events
    entry.bind("<KeyRelease>", lambda e: check_input(e, entry_var, data_list, listbox))
    listbox.bind("<<ListboxSelect>>", lambda e: fill_entry(e, entry_var, listbox))

create_autocomplete_search_bar(root, beers['beer_name'].tolist())

##Start the event loop
root.mainloop()