"""
    Firstly, we should convert train&test data.
"""
import pandas as pd

target = {
    'train': 'train',
    'test': 'test',
}

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for k, v in target.items():
    df = pd.read_csv('./data/input/' + k + '.' + extension, encoding="utf-8")
    '''
    if k == "train":
        print(df[(df[['x', 'y', 'z']] == 0).any(axis=1)].shape)
        df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)].reset_index(drop=True)
    '''
    df.to_feather('./data/interim/' + v + '.feather')
