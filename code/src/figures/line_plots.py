from plot_figure import FigurePlot
from src.models.dimensionality_reduction.model_dimensionality_reduction import modelPCA, modelTSNE, modelUMAP
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PlotSeries5axis(FigurePlot):
    def __init__(self, model_params, features, setting):
        super().__init__(model_params, features, setting)
        self.col = self.params.get('col')
        self.row = self.params.get('row')
        self.x = self.params.get('x')
        self.y = self.params.get('y')
        self.z = self.params.get('z')
        self.is_xlim = self.params.get('is_xlim')
        self.is_ylim = self.params.get('is_ylim')
        estimator = self.params.get('estimator')
        self.estimator = 'mean' if estimator is None else estimator
        ci = self.params.get('ci')
        self.ci = 95 if ci is None else ci

        self.run_name = '-'.join(filter(None, [self.col, self.row, self.x, self.y, self.z])) + self.run_name
        self.data = self.train

    def create_figure(self):
        col_categorical_uniques = self.data[self.col].unique()
        row_categorical_uniques = self.data[self.row].unique()
        n_col = len(col_categorical_uniques)
        n_row = len(row_categorical_uniques)
        _, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
        for i, col_v in tqdm(enumerate(col_categorical_uniques)):
            data = self.data.loc[self.data[self.col]==col_v, :]
            for j, row_v in enumerate(data[self.row].unique()):
                sns.lineplot(data=data.loc[data[self.row]==row_v, :], x=self.x, y=self.y, hue=self.z, ax=axes[j, i], estimator=self.estimator, ci=self.ci)
                axes[j, i].set_title(f'{self.col}: {col_v}, {self.row}: {row_v}')
                axes[j, i].legend(loc='upper left')
                if self.is_xlim is True:
                    axes[j, i].set_xlim(self.data[self.x].min(), self.data[self.x].max())
                if self.is_ylim is True:
                    axes[j, i].set_ylim(self.data[self.y].min(), self.data[self.y].max())


class PlotSeries4axis(FigurePlot):
    def __init__(self, model_params, features, setting):
        super().__init__(model_params, features, setting)
        self.col = self.params.get('col')
        self.row = self.params.get('row')
        self.x = self.params.get('x')
        self.y = self.params.get('y')
        self.z = self.params.get('z')
        self.is_xlim = self.params.get('is_xlim')
        self.is_ylim = self.params.get('is_ylim')
        estimator = self.params.get('estimator')
        self.estimator = 'mean' if estimator is None else estimator
        ci = self.params.get('ci')
        self.ci = 95 if ci is None else ci

        self.run_name = '-'.join(filter(None, [self.col, self.row, self.x, self.y, self.z])) + self.run_name
        self.data = self.train
    
    def create_figure(self):
        col_categorical_uniques = self.data[self.col].unique()
        n_col = len(col_categorical_uniques)
        _, axes = plt.subplots(1, n_col, figsize=(n_col*5, 5))

        for i, col_v in tqdm(enumerate(col_categorical_uniques)):
            data = self.train.loc[self.train[self.col]==col_v, :]
            axes[i].set_title(f'{self.col}: {col_v}')
            sns.lineplot(data=data, x=self.x, y=self.y, hue=self.z, ax=axes[i], estimator=self.estimator, ci=self.ci)
            axes[i].legend(loc='upper left')
            if self.is_xlim is True:
                axes[i].set_xlim(self.train[self.x].min(), self.train[self.x].max())
            if self.is_ylim is True:
                axes[i].set_xlim(self.train[self.y].min(), self.train[self.y].max())


class PlotSeries3axis(FigurePlot):
    def __init__(self, model_params, features, setting):
        super.__init__(model_params, features, setting)
        self.x = self.params.get('x')
        self.y = self.params.get('y')
        self.z = self.params.get('z')
        self.is_xlim = self.params.get('is_xlim')
        self.is_ylim = self.params.get('is_ylim')
        estimator = self.params.get('estimator')
        self.estimator = 'mean' if estimator is None else estimator
        ci = self.params.get('ci')
        self.ci = 95 if ci is None else ci

        self.run_name = '-'.join(filter(None, [self.x, self.y, self.z])) + self.run_name
        self.data = self.train

    def create_figure(self):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.lineplot(data=self.data, x=self.x, y=self.y, hue=self.z, ax=ax, estimator=self.estimator, ci=self.ci)
        ax.legend(loc='upper left')
        if self.is_xlim is True:
            ax.set_xlim(self.data[self.x].min(), self.data[self.x].max())
        if self.is_ylim is True:
            ax.set_ylim(self.data[self.y].min(), self.data[self.y].max())


class PlotSeries3axisMultipleFigure(FigurePlot):
    def __init__(self, model_params, features, setting):
        super.__init__(model_params, features, setting)
        self.x = self.params.get('x')
        self.y = self.params.get('y')
        self.z = self.params.get('z')
        self.is_xlim = self.params.get('is_xlim')
        self.is_ylim = self.params.get('is_ylim')
        estimator = self.params.get('estimator')
        self.estimator = 'mean' if estimator is None else estimator
        ci = self.params.get('ci')
        self.ci = 95 if ci is None else ci

        self.run_name = '-'.join(filter(None, [self.x, self.y, self.z])) + self.run_name
        self.data = self.train

    def create_figure(self, params):
        z_uniques = self.data[self.z].unique()
        n_z = len(z_uniques)
        _, axes = plt.subplots(1, n_z, figsize=(5*n_z, 5))
        for i, v in enumerate(z_uniques):
            sns.lineplot(data=self.data.loc[self.data[self.z]==v, :], x=self.x, y=self.y, ax=axes[i], estimator=self.estimator, ci=self.ci)
            axes[i].legend(loc='upper left')
            if self.is_xlim is True:
                axes[i].set_xlim(self.data[self.x].min(), self.data[self.x].max())
            if self.is_ylim is True:
                axes[i].set_ylim(self.data[self.y].min(), self.data[self.y].max())
            plt.show()