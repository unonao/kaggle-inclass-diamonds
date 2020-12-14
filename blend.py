import numpy as np
import pandas as pd
ID_name = "Id"
target_name = "SalePrice"

sub = pd.DataFrame(pd.read_feather(f'data/interim/test.feather')[ID_name])
sub[target_name] = 0
'''
weight = [0.60, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0]
print(sum(weight))
base_subs = {
    "stack": "data/output/sub_stacking_20201125102411_0.1227882374874322.csv",
    "cat": "data/output/sub_CatBoost_20201125121612_0.12094692517779954.csv",
    "gb": "data/output/sub_GradientBoosting_20201125082920_0.12171807998255628.csv",
    "lgbm": "data/output/sub_LightGBM_20201125144925_0.12364890239035035.csv",
    "xgb": "data/output/sub_XGBoost_20201124235212_0.12438152375202484.csv",
    "lasso": "data/output/sub_Lasso_20201125122436_0.1258938527175052.csv",
    "elastic": "data/output/sub_ElasticNet_20201125122605_0.12613411927853602.csv",
    "kr": "data/output/sub_KernelRidge_20201126213738_0.12741842026803182.csv",
    "ridge": "data/output/sub_Ridge_20201126213756_0.1279679743707424.csv",
    "rf": "data/output/sub_RandomForest_20201126213814_0.14201761429984638.csv",
}

for n, (base, path) in enumerate(base_subs.items()):
    tmp_sub = pd.read_csv(path)
    sub[target_name] += tmp_sub[target_name] * weight[n]
'''


base_subs = {
    "cat": "data/output/sub_CatBoost_20201125121612_0.12094692517779954.csv",
    "elastic": "data/output/sub_ElasticNet_20201125122605_0.12613411927853602.csv",
    # "gb": "data/output/sub_GradientBoosting_20201122182203_0.12691619533179993.csv",
    "lasso": "data/output/sub_Lasso_20201125122436_0.1258938527175052.csv",
    "lgbm": "data/output/sub_LightGBM_20201125145255_0.1235315066189184.csv",
    # "xgb": "data/output/sub_XGBoost_20201122182607_0.12749830372143028.csv",
    "stack": "data/output/sub_stacking_20201125102411_0.1227882374874322.csv",
    #    "rf": "data/output/sub_RandomForest_20201122182446_0.14382690434655043.csv",
    #    "kr": "data/output/sub_KernelRidge_20201122182356_0.1297173732291673.csv",
    #   "ridge": "data/output/sub_Ridge_20201122182514_0.129362827976269.csv",
}

for base, path in base_subs.items():
    tmp_sub = pd.read_csv(path)
    if base == "stack":
        sub[target_name] += tmp_sub[target_name] * 6
    else:
        sub[target_name] += tmp_sub[target_name]
sub[target_name] /= 10


sub.to_csv(
    './data/output/sub_blend.csv',
    index=False
)

'''

'''
