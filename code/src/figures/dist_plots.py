from plot_figure import FigurePlot
from src.models.dimensionality_reduction.model_dimensionality_reduction import modelPCA, modelTSNE, modelUMAP
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PlotDist2axis(FigurePlot):
    def __init__(self, model_params, features, setting):
        super().__init__(model_params, features, setting)
        self.col = self.params.get('col')
        self.row = self.params.get('row')
        self.target = self.params.get('target')

        self.run_name = '-'.join(filter(None, [self.col, self.row])) + self.run_name
        self.data = self.train

    def create_figure(self):
        col_categorical_uniques = self.data[self.col].unique()
        row_categorical_uniques = self.data[self.row].unique()
        n_col = len(col_categorical_uniques)
        n_row = len(row_categorical_uniques)
        _, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
        
        for i, col_v in tqdm(enumerate(col_categorical_uniques)):
            for j, row_v in enumerate(self.data[self.row].unique()):
                data = self.data.loc[(self.data[self.col]==col_v) & (self.data[self.row]==row_v), :]
                sns.distplot(data[self.target], ax=axes[j, i])
                axes[j, i].set_title(f'{self.col}={col_v}, {self.row}={row_v}')


class PlotDist1axis(FigurePlot):
    def __init__(self, model_params, features, setting):
        super().__init__(model_params, features, setting)
        self.col = self.params.get('col')
        self.row = self.params.get('row')
        self.target = self.params.get('target')

        self.run_name = '-'.join(filter(None, [self.col, self.row])) + self.run_name
        self.data = self.train

    def create_figure(self):
        col_categorical_uniques = self.data[self.col].unique()
        n_col = len(col_categorical_uniques)
        _, axes = plt.subplots(1, n_col, figsize=(n_col*5, 5))
        
        for i, col_v in tqdm(enumerate(col_categorical_uniques)):
            data = self.data.loc[(self.data[self.col]==col_v), :]
            sns.distplot(data[self.target], ax=axes[i])
            axes[i].set_title(f'{self.col}={col_v}')