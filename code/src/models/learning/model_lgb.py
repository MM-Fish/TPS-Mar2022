from dataclasses import asdict
from asyncio.log import logger
from ctypes import util
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
import lightgbm as lgb
import optuna.integration.lightgbm as opt_lgb
from model import Model
from util import Util
import json

# 各foldのモデルを保存する配列
model_array = []
eval_results_array = []
best_params_array = []

class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, tr_y)

        if validation:
            dvalid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = asdict(self.params)
        num_round = params.pop('num_round')
        verbose = params.pop('verbose')
        optuna = params.pop('optuna')

        # 学習
        if validation:
            eval_results = {}
            early_stopping_rounds = params.pop('early_stopping_rounds')
            if optuna:
                self.model = opt_lgb.train(
                                    params,
                                    dtrain,
                                    num_boost_round=num_round,
                                    valid_sets=(dtrain, dvalid),
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose_eval=verbose,
                                    evals_result=eval_results
                                    )
                best_params_array.append(self.model.params)
            else:
                self.model = lgb.train(
                                    params,
                                    dtrain,
                                    num_boost_round=num_round,
                                    valid_sets=(dtrain, dvalid),
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose_eval=verbose,
                                    evals_result=eval_results
                                    )
            model_array.append(self.model)
            eval_results_array.append(eval_results)

        else:
            self.model = lgb.train(params, dtrain, num_boost_round=num_round)
            model_array.append(self.model)


    # shapを計算しないver
    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


    # shapを計算するver
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        return valid_prediticion, fold_importance


    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)


    @classmethod
    def calc_feature_importance(self, dir_name, run_name, features):
        """feature importanceの計算
        """

        val_split = model_array[0].feature_importance(importance_type='split')
        val_gain = model_array[0].feature_importance(importance_type='gain')
        val_split = pd.Series(val_split)
        val_gain = pd.Series(val_gain)

        # -----------
        # splitの計算
        # -----------
        if len(model_array) > 1:
            for m in model_array[1:]:
                s = pd.Series(m.feature_importance(importance_type='split'))
                val_split = pd.concat([val_split, s], axis=1)
                s = pd.Series(m.feature_importance(importance_type='gain'))
                val_gain = pd.concat([val_gain, s], axis=1)
            # 各foldの平均と標準偏差を算出
            val_mean = val_split.mean(axis=1)
            val_std = val_split.std(axis=1)
        else:
            val_mean = val_split
            val_std = val_split
        val_mean = val_mean.values
        importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')
        val_std = val_std.values
        importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10, 10))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance split')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
        ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        #凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.93), loc='upper right', borderaxespad=0.5, fontsize=12)

        #グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        plt.savefig(dir_name + run_name + '_fi_split.png', dpi=300, bbox_inches="tight")
        plt.close()

        # -----------
        # gainの計算
        # -----------
        if len(model_array) > 1:
            for m in model_array[1:]:
                s = pd.Series(m.feature_importance(importance_type='split'))
                val_split = pd.concat([val_split, s], axis=1)
                s = pd.Series(m.feature_importance(importance_type='gain'))
                val_gain = pd.concat([val_gain, s], axis=1)
            # 各foldの平均と標準偏差を算出
            val_mean = val_gain.mean(axis=1)
            val_std = val_gain.std(axis=1)
        else:
            val_mean = val_split
            val_std = val_split

        val_mean = val_mean.values
        importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')
        val_std = val_std.values
        importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

        # 変動係数を算出
        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10, 10))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
        ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.93), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        plt.savefig(dir_name + run_name + '_fi_gain.png', dpi=300, bbox_inches="tight")
        plt.close()

    @classmethod
    def calc_loss_curve(self, dir_name, run_name):
        # 損失推移を表示
        i = 0
        n_fold = len(eval_results_array)
        if n_fold > 1:
            _, axes = plt.subplots(n_fold, 1, figsize=(10, n_fold*5))
            for i in range(n_fold):
                loss_func_name = list(eval_results_array[i]['training'].keys())[0]
                loss_train = eval_results_array[i]['training'][loss_func_name]
                loss_test = eval_results_array[i]['valid_1'][loss_func_name]
                
                axes[i].plot(loss_train, label='train loss')
                axes[i].plot(loss_test, label='test loss')
                axes[i].set_title(f"fold:{i}")
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel('logloss')
                axes[i].legend()
        else:
            ax = plt.subplot()
            loss_func_name = list(eval_results_array[i]['training'].keys())[0]
            loss_train = eval_results_array[i]['training'][loss_func_name]
            loss_test = eval_results_array[i]['valid_1'][loss_func_name]
            
            ax.plot(loss_train, label='train loss')
            ax.plot(loss_test, label='valid loss')
            ax.set_title(f"fold:{i}")
            ax.set_xlabel('Iteration')
            ax.set_ylabel(loss_func_name)
            ax.legend()            
        plt.savefig(dir_name + run_name + '_loss_curve.png', dpi=300, bbox_inches="tight")
        plt.close()

    def save_optuna_best_params(self, dir_name, run_name):
        '''
        '''
        eval_results_dict = {}
        for i, v in enumerate(eval_results_array):
            eval_results_dict[i] = v
        fw = open(dir_name + run_name  + '_param.json', 'w')
        json.dump(eval_results_dict, fw, indent=4, default=self.set_default)

    def set_default(self, obj):
        """json出力の際にset型のオブジェクトをリストに変更する
        """
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
