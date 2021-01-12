import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plugins to plot
#TODO histogram
#

def generate_heatmap(df):
    corr = df.corr()

    plt.figure(figsize=(9,7))
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        linecolor='white',
        linewidths=0.1,
        cmap="RdBu"
    )
    plt.show()

def generate_barplot(x, y, data, order=None):
    plt.figure(figsize=(9,7))
    sns.set_theme(style="whitegrid")
    # plt.xticks(y_pos, bars, color='orange', rotation=45, fontweight='bold', fontsize='17', horizontalalignment='right')

    ax = sns.barplot(x=x, y=y, data=data, order=order)

    plt.show()

def plot_cat_levels(series):
   df = pd.DataFrame(series.value_counts()).reset_index()
   generate_barplot(df['index'], df.iloc[:,1]/ df.iloc[:,1].sum(), df)

