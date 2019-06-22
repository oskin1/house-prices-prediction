import os
from utils import TabularDataset, fix_missing
from models import FFNN
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

df = pd.read_csv(train_file_path)

target_feature = 'SalePrice'

df[target_feature].describe()

corr_mat = df.corr()

num_features = 11
cols = corr_mat.nlargest(num_features, target_feature)[target_feature].index
new_corr_mat = np.corrcoef(df[cols].values.T)

num_missing_values = df.isnull().sum().sort_values(ascending=False)
percent_missing = ((df.isnull().sum() * 100) / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([num_missing_values, percent_missing], axis=1, keys=['Num', '%'])
missing_data.head(20)

df = df.drop(missing_data[missing_data['%'] > 1].index, axis=1)
# Missing values left
df.isnull().sum().max()

df = df.drop(['Id', 'Utilities'], axis=1)
# All features left
features = df.columns

cat_features = [
    "MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt", 'YearRemodAdd', 'RoofStyle',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'YrSold', 'SaleType', 'SaleCondition',
    'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood'
]

save_dir = 'data/'
training_state_file = 'train_state_v2.pt'

num_epochs = 1000
batch_size = 512


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    try:
        checkpoint = torch.load(os.path.join(save_dir, training_state_file))
    except FileNotFoundError:
        checkpoint = None

    cont_features = [feature for feature in features if feature not in cat_features + [target_feature]]

    for feature in cont_features:
        fix_missing(df, df[feature], feature, {})

    data = df.dropna(axis=0)

    label_encoders = {}
    for cat_col in cat_features:
        label_encoders[cat_col] = LabelEncoder()
        data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

    dataset = TabularDataset(data, cat_cols=cat_features, output_col=target_feature)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)

    cat_dims = [int(data[col].nunique()) for col in cat_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    model = FFNN(emb_dims, num_cont_features=31, lin_layer_sizes=[100, 500, 500],
                 output_size=1, emb_dropout=.04, lin_layer_dropouts=[.001, .01, .01]).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)

    epochs_done = 0

    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        epochs_done = checkpoint['epochs_done']

    for epoch in range(num_epochs - epochs_done):

        running_loss = .0

        for y, cont_x, cat_x in data_loader:
            y = y.to(device)
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)

            predictions = model(cont_x, cat_x)
            loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 100 == 0:
            torch.save({
                'epochs_done': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_dir, training_state_file))

        print(f'Epoch {epoch + epochs_done}, running loss {np.sqrt(running_loss)}')
