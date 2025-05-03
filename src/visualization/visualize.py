"""
Functions to visualize music data and model results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_music_features(data, features, save_path=None):
    """
    Plot distributions of music features.
    
    Args:
        data (pandas.DataFrame): DataFrame containing music features
        features (list): List of feature names to plot
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 3*len(features)))
    
    if len(features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        if feature in data.columns:
            sns.histplot(data[feature], ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
        else:
            axes[i].text(0.5, 0.5, f"Feature '{feature}' not found in data", 
                         horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_performance(metrics, save_path=None):
    """
    Plot model performance metrics.
    
    Args:
        metrics (dict): Dictionary of metric names and values
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 