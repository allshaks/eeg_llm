# %%
import numpy as np 
import matplotlib.pyplot as plt 
import mne 
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from torch.optim import Adam 
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# %%
# load data and epoch it 
raw = mne.io.read_raw_eeglab('sep_uwgr_prepro.set', preload=False)
events, event_id = mne.events_from_annotations(raw)

tmin = -0.05
tmax = 0.22

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=False)

# %%
# get epoched data and average data 
data_original = epochs.get_data()
avg_data_original = epochs.average().data

# %%
# reshape the data according to model input 
data = np.transpose(data_original, (0, 2, 1))
avg_data = np.transpose(avg_data_original, (1, 0))

# %%
# plot a channel of a trial of the raw data 
tr = 30
ch = 42
plt.figure(figsize=(10, 5))
plt.plot(data[tr, :, ch], label=f"{ch+1}th channel of {tr+1}th trial: raw data")
plt.legend()
plt.show()

# %%
# plot a channel of the average data 
ch = 42
plt.figure(figsize=(10, 5))
plt.plot(avg_data[:, ch], label=f"{ch+1}th channel: average data")
plt.legend()
plt.show()

# %%
# plot the mean global field power 


# %%
def split_data(data, test_size=0.2, random_state=None):
    X_train, y = train_test_split(data, test_size=test_size, random_state=random_state)
    return X_train, y

# %%
# split the data into training and test set 
X_train, y = split_data(data, random_state=42)

# %%
def create_data_loaders(X_train, X_test, batch_size=16, shuffle=True):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train)
    test_dataset = TensorDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader 

# %%
# batch the data 
train_loader, test_loader = create_data_loaders(X_train, y)

# %%
def split_past_future(batch, num_future_points=None):
    if len(batch.shape) == 3:
        past_values = batch[:, :-num_future_points, :]
        future_values = batch[:, -num_future_points:, :]
    else: 
        past_values = batch[:, :-num_future_points]
        future_values = batch[:, -num_future_points:]
    return past_values, future_values

# %%
# get raw data for one channel for testing purposes 
ch = 42
ch_data = data[:, :, ch]
ch_train, ch_test = split_data(ch_data, random_state=42)
ch_train_loader, ch_test_loader = create_data_loaders(ch_train, ch_test)


# %%
# define the configuration of the model 
ch_config = TimeSeriesTransformerConfig(
    prediction_length=154,  # length of the future values to predict
    context_length=399,  # length of the past values to use
    num_time_features=1, # number of time features
    encoder_layers=2,  # number of transformer layers in the encoder
    decoder_layers=2,  # number of transformer layers in the decoder
    d_model=32,  # dimension of the model
    n_heads=4,  # number of attention heads
    input_size=1, # size of the input 
    lags_sequence=[1], # sequence of lags 
    distribution_output='normal', # distribution where the output is sampled from
    )

# initialize the model 
model = TimeSeriesTransformerForPrediction(ch_config)

# %%
# define the prediciton lenght
num_future_points = 154
# Initialize optimizer
optim = Adam(model.parameters(), lr=1e-3)
# store the params 
params_lst = []
# Define the number of epochs 
num_epochs = 20

# iterate over all epochs 
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0 
    epoch_params = []

    # iterate over each batch 
    for batch in ch_train_loader: 
        past_values, future_values = split_past_future(batch[0], num_future_points=num_future_points)
        batch_dict = {
            "past_values": past_values,  # (batch_size, input_length, input_size)
            "future_values": future_values,  # (batch_size, prediction_length, input_size)
            "past_time_features": torch.arange(past_values.size(1)).unsqueeze(0).unsqueeze(2).float().repeat(past_values.size(0), 1, 1),  # (batch_size, seq_length, 1)
            "past_observed_mask": torch.ones_like(past_values),  # (batch_size, seq_length, input_size)
            "future_observed_mask": torch.ones_like(future_values), # (batch_size, prediciton_lenght, input_size)
            "future_time_features": torch.arange(past_values.size(1), past_values.size(1) + num_future_points).unsqueeze(0).unsqueeze(2).float().repeat(future_values.size(0), 1, 1),  # (batch_size, prediction_length, 1)
            "return_dict": True
        }

        # Forward pass
        outputs = model(
            past_values=batch_dict["past_values"],
            past_time_features=batch_dict["past_time_features"],
            past_observed_mask=batch_dict["past_observed_mask"],
            future_observed_mask=batch_dict["future_observed_mask"],
            future_values=batch_dict["future_values"],
            future_time_features=batch_dict["future_time_features"],
            return_dict=batch_dict["return_dict"]
        )

        loss = outputs.loss
        scale = outputs.scale
        loc = outputs.loc

        params = (scale * outputs.params[0] + loc, scale * outputs.params[1])
        epoch_params.append(params)

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
    params_lst.append(epoch_params)
    # params_lst.append(epoch_params)
    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %%



