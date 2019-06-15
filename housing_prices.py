import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from models import FFNN
from utils import TabularDataset, fix_missing

batch_size = 512
train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

save_dir = 'data/'
training_state_file = 'train_state.pt'

features = [
    "SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "YearBuilt",
    "LotShape", "1stFlrSF", "2ndFlrSF", 'YearRemodAdd', 'RoofStyle', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
]

cat_features = [
    "MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt", 'YearRemodAdd', 'RoofStyle',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'YrSold', 'SaleType', 'SaleCondition'
]

# Prediction target
target_feature = 'SalePrice'

num_epochs = 5000


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    try:
        checkpoint = torch.load(os.path.join(save_dir, training_state_file))
    except FileNotFoundError:
        checkpoint = None

    data = pd.read_csv(train_file_path, usecols=features)

    cont_features = [feature for feature in features if feature not in cat_features + [target_feature]]

    for feature in cont_features:
        fix_missing(data, data[feature], feature, {})

    data = data.dropna(axis=0)

    label_encoders = {}
    for cat_col in cat_features:
        label_encoders[cat_col] = LabelEncoder()
        data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

    dataset = TabularDataset(data, cat_cols=cat_features, output_col=target_feature)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)

    cat_dims = [int(data[col].nunique()) for col in cat_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    model = FFNN(emb_dims, num_cont_features=33, lin_layer_sizes=[500, 1000, 1000],
                 output_size=1, emb_dropout=.04, lin_layer_dropouts=[.001, .01, .01]).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

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
            loss = torch.sqrt(loss_fn(predictions, y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

        if epoch % 500 == 0:
            torch.save({
                'epochs_done': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_dir, training_state_file))

        print(f'Epoch {epoch + epochs_done}, running loss {running_loss}')

    print("Done")
