import os
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# pretty color cycles & settings
def color_cycles():
    color1 = ['#68C469','#68C497','#68C3C4','#6895C4','#6968C4','#9768C4']
    color2 = ['#B468C4','#C468BD','#C468A6','#C4688F','#C46878','#C46F68']
    color3 = ['#C8AC4F','#C6C84F','#A8C84F','#89C84F','#4FC84F','#4FC86D']
    return color1, color2, color3
mpl.rcParams['font.size'] = 8

# pie chart
def pie_chart(df, column, title='pie_chart', filename='pie_chart', colors=['#3EA607', '#5F9343', '#868686', '#93435F', '#A6073E']):
    series = df[column].dropna()
    pie, ax = plt.subplots(figsize=(10,4))
    plt.pie(x=series, autopct="%1.1f%%", explode= [0.02]*series.shape[0], labels=series.keys(), colors=colors)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    pie.savefig(f"results/{filename}.png")
    return pie, ax

# horizontal bar chart
def horizontal_bars(df, column, percentage=True, xlabel='', title='bar_plot', filename='bar_plot'):
    # sort descending by column value
    df = df.sort_values(column, ascending=False)
    if percentage == True:
        df = df*100
    # plot
    bar, ax = plt.subplots(figsize=(10,4))
    # ax.xaxis.grid(True, linestyle='--', linewidth=.7, color='#B7B7B7')
    # ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    bars = ax.barh(df.index, width=df[column], height=0.9, align='center', color='#739D81')
    for b in bars:
        height = b.get_height()
        width = b.get_width()
        position = b.get_y()
        ax.annotate('{}'.format(round(width, 1)), xy=(width/2, height/2 + position), va='center', fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    bar.savefig(f"results/{filename}.png")
    return bar, ax


