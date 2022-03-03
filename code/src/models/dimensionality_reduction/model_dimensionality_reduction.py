import pandas as pd
import numpy as np
from typing import Callable, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP as UMAP
from matplotlib import pyplot as plt
import seaborn as sns

# ##PCA
class modelPCA():
    def __init__(self, model_params):
        self.params = model_params
        
    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        self.train_size = len(train)

        self.model = PCA()
        transformed_data = self.model.fit_transform(all_data.select_dtypes(exclude='object'))
        self.accum_contribution_rate = np.cumsum(self.model.explained_variance_ratio_)
        return transformed_data
    
    def plotfig(self, x: pd.DataFrame, y: pd.DataFrame, target: np.ndarray, dir_name: str, run_name: str):
        # グラフ化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x, y, alpha=0.8, hue=target, ax=axes[0])
        sns.lineplot([n for n in range(1, len(self.accum_contribution_rate)+1)], self.accum_contribution_rate, markers=True, ax=axes[1])
        plt.savefig(dir_name + f'{run_name}.png', dpi=300, bbox_inches="tight")
        plt.close()

    def processing_data(self, transformed_data):
        processed_train = pd.DataFrame(transformed_data).iloc[:self.train_size, self.accum_contribution_rate <= self.params['thres']]
        processed_test = pd.DataFrame(transformed_data).iloc[self.train_size:, self.accum_contribution_rate <= self.params['thres']]
        return processed_train, processed_test

# ##UMAP
class modelUMAP():
    def __init__(self, model_params):
        self.params = model_params
        
    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        self.train_size = len(train)

        self.model = UMAP(n_components=self.params['n_components'], n_neighbors=self.params['n_neighbors'])
        return self.model.fit_transform(all_data.select_dtypes(exclude='object'))
    
    def plotfig(self, x: pd.DataFrame, y: pd.DataFrame, target: np.ndarray, dir_name: str, run_name: str):
        # グラフ化
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x, y, alpha=0.8, hue=target, ax=ax)
        plt.savefig(dir_name + f'{run_name}.png', dpi=300, bbox_inches="tight")
        plt.close()
    
    def processing_data(self, transformed_data):
        processed_train = pd.DataFrame(transformed_data).iloc[:self.train_size, :]
        processed_test = pd.DataFrame(transformed_data).iloc[self.train_size:, :]
        return processed_train, processed_test


# ##t-SNE
class modelTSNE():
    def __init__(self, model_params):
        self.params = model_params
        
    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        self.train_size = len(train)

        self.model = TSNE(n_components=self.params['n_components'], perplexity=self.params['perplexity'])
        return self.model.fit_transform(all_data.select_dtypes(exclude='object'))
    
    # それぞれのモデルによって出力したいグラフが変わるので、DimensionalityReductionではなくmodelのメソッドにする。
    def plotfig(self, x: pd.DataFrame, y: pd.DataFrame, target: np.ndarray, dir_name: str, run_name: str):
        # グラフ化
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x, y, alpha=0.8, hue=target, ax=ax)
        plt.savefig(dir_name + f'{run_name}.png', dpi=300, bbox_inches="tight")
        plt.close()
    
    def processing_data(self, transformed_data):
        processed_train = pd.DataFrame(transformed_data).iloc[:self.train_size, :]
        processed_test = pd.DataFrame(transformed_data).iloc[self.train_size:, :]
        return processed_train, processed_test