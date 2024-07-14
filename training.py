import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pricing_model import PricingModel
from data import OptionsDataset, pull_options_data, get_all_options_data
import joblib
from yahoo_fin import stock_info as si
import pandas as pd



options_data = get_all_options_data()

options_data = options_data.drop_duplicates().dropna()

volume_threshold = 1000
options_data = options_data[options_data['volume'] >= volume_threshold].reset_index(drop=True)

print(options_data)

scaler = StandardScaler()

feature_columns = ['strike', 'historicalVolatility', 'timeToMaturity', 'moneyness', 'optionType', 'riskFreeRate']
features = options_data[feature_columns]
features = scaler.fit_transform(features)

targets = ((options_data['bid'] + options_data['ask']) / 2).values

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.05, random_state=1)

train_dataset = OptionsDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = OptionsDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

pricing_model = PricingModel(len(feature_columns))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(pricing_model.parameters(), lr=0.0001)


num_epochs = 300
for epoch in range(num_epochs):
    pricing_model.train()

    running_loss = 0.0
    for features_batch, target_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = pricing_model(features_batch)

        loss = criterion(torch.flatten(outputs), target_batch)
        loss.backward()

        optimizer.step()

        running_loss =+ loss.item()

    print(f"Epoch {epoch + 1} loss: {running_loss / len(train_dataloader)}")

torch.save(pricing_model.state_dict(), "options_pricing_model.pth")

joblib.dump(scaler, "scaler.pkl")

pricing_model.eval()
test_loss = 0.0
with torch.no_grad():
    for features_batch, target_batch in test_dataloader:
        outputs = pricing_model(features_batch)
        print(outputs.item(), target_batch.item())
        loss = criterion(torch.flatten(outputs), target_batch)
        test_loss += loss.item()

print(f"Test average MSE loss: {test_loss / len(test_dataloader)}")

