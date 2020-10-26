import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_random_colors(n_colors):
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_colors))
    return colors

def is_even(number):
    return True if number % 2 == 0 else False

def binary_operator(number, first_value, second_value):
    return first_value if is_even(number) else second_value

def main(args):
    df = pd.read_csv(file_path)
    df = df.groupby(['WHO Region']).sum().reset_index()
    continents = df['WHO Region'].values
    loading_vectors_name = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered']
    data = df[loading_vectors_name].values
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data_reduced_dimension = pca.fit_transform(data)
    loading_vectors = pca.components_.T

    final_df = pd.DataFrame(data=data_reduced_dimension , columns=['Phi-1', 'Phi-2'])
    final_df = pd.concat([final_df, df], axis = 1)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].set_xlabel('Phi-1', fontsize=12)
    ax[0].set_ylabel('Phi-2', fontsize=12)
    ax[0].set_title('Covid-19 over continents', fontsize = 15)
    colors = get_random_colors(len(continents))
    for index, continent in enumerate(continents):
        ax[0].scatter(
            final_df.loc[final_df['WHO Region'] == continent, 'Phi-1'], 
            final_df.loc[final_df['WHO Region'] == continent, 'Phi-2'], 
            c = np.array([colors[index]]), 
            s = 100
        )

    ax[0].legend(continents)
    ax[0].grid()
    ax[1].set_xlabel('x', fontsize=12)
    ax[1].set_ylabel('y', fontsize=12)
    ax[1].set_title('loading vectors',  x=0.9, y=0.9, fontsize = 15)
    for i in range(len(loading_vectors_name)): 
        plt.text(
            loading_vectors[i, 0]+0.1, 
            binary_operator(i, loading_vectors[i, 1], loading_vectors[i, 1]-0.1), 
            loading_vectors_name[i]
        )
        ax[1].arrow(
            0, 0, 
            loading_vectors[i, 0], 
            loading_vectors[i, 1], 
            head_width=0.1, 
            head_length=0.1, 
            fc=binary_operator(i, 'blue', 'red'), 
            ec=binary_operator(i, 'black', 'gray')
        )

    plt.xlim(-.2,.8)
    plt.ylim(-1,1)
    plt.draw()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unsupervised learning covid-19')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, './data/full_grouped.csv')
    parser.add_argument('--file_path', type=str, default=file_path, metavar='path', help='path of data file')
    args = parser.parse_args()
    main(args)
