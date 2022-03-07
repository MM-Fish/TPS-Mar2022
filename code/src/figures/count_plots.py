from plot_figure import FigurePlot
from src.models.dimensionality_reduction.model_dimensionality_reduction import modelPCA, modelTSNE, modelUMAP
import matplotlib.pyplot as plt
import seaborn as sns

class PlotCount(FigurePlot):
    def __init__(self, model_params, features, setting):
        super().__init__(model_params, features, setting)
        self.cols = self.params.get('cols')
        self.target = self.params.get('target')

        self.run_name = 'counts'

    def create_figure(self):
        fig, axes = plt.subplots(len(self.cols), 2, figsize=(10, len(self.cols)*5))
        for i, col in enumerate(self.cols):
            sns.countplot(x=col, data = self.train, ax=axes[i, 0])
            if col != self.target:
                sns.countplot(x=col, data = self.test, ax=axes[i, 1])