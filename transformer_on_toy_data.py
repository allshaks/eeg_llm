# %%
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import numpy as np 
from torch.optim import Adam 
import matplotlib.pyplot as plt 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 

# %%
def generate_wave_data(num_samples=600, seq_length=100, noise=0):
    data = []

    phase_shift_list = np.empty([0])

    for _ in range(num_samples):
        phase_shift = np.random.uniform(-20, 20)
        w = 1 # + np.random.randn(1)*0.01
        t = np.linspace(0, 4*np.pi, seq_length) + phase_shift 

        phase_shift_list = np.append(phase_shift_list, phase_shift)

        sine = np.sin(w*t) + noise * np.random.randn(seq_length)
        cosine = np.cos(w*t) + noise * np.random.randn(seq_length)

        sample = np.stack([sine, cosine], axis=1)
        data.append(sample)
    data = np.array(data)
    return data, phase_shift_list

# %%
def gaussian(t, mean, std):
    return 1/np.sqrt(2*np.pi*std**2)*np.exp(-(t-mean)**2/(2*std**2))

# %%
def generate_pulse_data(num_samples=6000, seq_length=100, noise=1):
    data = []

    phase_shift_list = np.empty([0])

    for _ in range(num_samples):
        phase_shift = np.random.randn(1)
        w = 1 # + np.random.randn(1)*0.01
        t = np.linspace(0, 5, seq_length) + phase_shift

        phase_shift_list = np.append(phase_shift_list, phase_shift) 
        pulse1 = gaussian(t, 1, 0.3) + noise * np.random.randn(seq_length)
        pulse2 = gaussian(t, 2, 0.3) + noise * np.random.randn(seq_length)

        sample = np.stack([pulse1, pulse2], axis=1)
        data.append(sample)
    data = np.array(data)
    return data, phase_shift_list

# %%
# generate toy data  
# data, _ = generate_wave_data()
data, _ = generate_pulse_data()

# %%
# plot a sample to inspect the level of noise 
s = 30
plt.figure(figsize=(10, 5))
plt.plot(data[s, :, 0], label="Noisy Sine")
plt.plot(data[s, :, 1], label="Noisy Cosine")
plt.legend()
plt.savefig("Noisy_wave_data.png")
plt.clf()

# %%
# plot the average to see that the clean signal can be recovered 
data_avg = np.mean(data, axis=0) 
plt.figure(figsize=(10, 5))
plt.plot(data_avg[:, 0], label="Sine")
plt.plot(data_avg[:, 1], label="Cosine")
plt.legend()
plt.savefig("Average.png")
plt.clf()

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
def split_past_future(batch, num_future_points=10):
    past_values = batch[:, :-num_future_points, :]
    future_values = batch[:, -num_future_points:, :]
    return past_values, future_values

# %%
# define the configuration of the model 
config = TimeSeriesTransformerConfig(
    prediction_length=70,  # length of the future values to predict
    context_length=29,  # length of the past values to use
    num_time_features=1, # number of time features
    encoder_layers=2,  # number of transformer layers in the encoder
    decoder_layers=2,  # number of transformer layers in the decoder
    d_model=32,  # dimension of the model
    n_heads=4,  # number of attention heads
    input_size=2, # size of the input 
    lags_sequence=[1], # sequence of lags 
    distribution_output='normal', # distribution where the output is sampled from
    )

# initialize the model 
model = TimeSeriesTransformerForPrediction(config)

# %%
# define the prediciton lenght
num_future_points = 70
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
    for batch in train_loader: 
        past_values, future_values = split_past_future(batch[0], num_future_points=num_future_points)

        batch = {
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
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            future_observed_mask=batch["future_observed_mask"],
            future_values=batch["future_values"],
            future_time_features=batch["future_time_features"],
            return_dict=batch["return_dict"]
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
epoch_dict = np.array([
    [{"mu_sine": 0, "sigma_sine": 0, "mu_cosine": 0, "sigma_cosine": 0}
     for _ in range(len(params_lst[0][0][0][0]))]
    for _ in range(len(params_lst))
])
num_batches = len(params_lst[0])


for i, epoch in enumerate(params_lst):
    #prediction_dict = np.full(len(params_lst[0][0][0][0]))

    for j, batch in enumerate(epoch): 
        mean = batch[0]
        std_dev = batch[1]

        for time_point in range(mean.size(1)):
            

            mu_sine = mean[:, time_point, 0].detach().numpy()
            sigma_sine = std_dev[:, time_point, 0].detach().numpy()
            mu_cosine = mean[:, time_point, 1].detach().numpy()
            sigma_cosine = std_dev[:, time_point, 1].detach().numpy()

            epoch_dict[i,time_point]["mu_sine"] += np.mean(mu_sine)
            epoch_dict[i,time_point]["sigma_sine"] += np.mean(sigma_sine)
            epoch_dict[i,time_point]["mu_cosine"] += np.mean(mu_cosine)
            epoch_dict[i,time_point]["sigma_cosine"] += np.mean(sigma_cosine)

    for time_point in range(len(epoch_dict[i])):
        epoch_dict[i,time_point]["mu_sine"] /= num_batches
        epoch_dict[i,time_point]["sigma_sine"] /= num_batches
        epoch_dict[i,time_point]["mu_cosine"] /= num_batches
        epoch_dict[i,time_point]["sigma_cosine"] /= num_batches


# %%
epoch_df = pd.DataFrame(epoch_dict.tolist())

# %%
time_point = 9
x_ep = [i for i in range(num_epochs)]
sine_mu_across_epochs = epoch_df[time_point].apply(lambda x: x["mu_sine"]).values 
sine_sigma_across_epochs = epoch_df[time_point].apply(lambda x: x["sigma_sine"]).values
cosine_mu_across_epochs = epoch_df[time_point].apply(lambda x: x["mu_cosine"]).values
cosine_sigma_across_epochs = epoch_df[time_point].apply(lambda x: x["sigma_cosine"]).values


# %%
plt.figure(figsize=(10, 5))
plt.plot(x_ep, sine_mu_across_epochs, label="Sine Mean", color='blue')
plt.plot(x_ep, cosine_mu_across_epochs, label="Cosine Mean", color="yellow")

plt.fill_between(x_ep, 
                 [m - s for m, s in zip(sine_mu_across_epochs, sine_sigma_across_epochs)],
                 [m + s for m, s in zip(sine_mu_across_epochs, sine_sigma_across_epochs)], 
                color='orange', 
                alpha=0.2,
)
plt.fill_between(x_ep, 
                 [m - s for m, s in zip(cosine_mu_across_epochs, cosine_sigma_across_epochs)],
                 [m + s for m, s in zip(cosine_mu_across_epochs, cosine_sigma_across_epochs)], 
                color='purple', 
                alpha=0.2,
)
plt.legend()
plt.savefig("Mu_across_epochs.png")
plt.clf()

# %%
ep = 19
samples = epoch_df.shape[1]
t = [i for i in range(samples)]
sine_mu_across_samples = epoch_df.loc[ep, :].apply(lambda x: x["mu_sine"]).values 
sine_sigma_across_samples = epoch_df.loc[ep, :].apply(lambda x: x["sigma_sine"]).values
cosine_mu_across_samples = epoch_df.loc[ep, :].apply(lambda x: x["mu_cosine"]).values
cosine_sigma_across_samples = epoch_df.loc[ep, :].apply(lambda x: x["sigma_sine"]).values

# %%
plt.figure(figsize=(10, 5))
plt.plot(t, sine_mu_across_samples, label="Sine Mean", color='blue')
plt.plot(t, cosine_mu_across_samples, label="Cosine Mean", color="yellow")

# # plt.fill_between(t, 
#                  [m - s for m, s in zip(sine_mu_across_samples, sine_sigma_across_samples)],
#                  [m + s for m, s in zip(sine_mu_across_samples, sine_sigma_across_samples)], 
#                 color='orange', 
#                 alpha=0.2,
# )
# # plt.fill_between(t, 
#                  [m - s for m, s in zip(cosine_mu_across_samples, cosine_sigma_across_samples)],
#                  [m + s for m, s in zip(cosine_mu_across_samples, cosine_sigma_across_samples)], 
#                 color='purple', 
#                 alpha=0.2,
# )
plt.legend()
plt.savefig("Mu_across_samples.png")
plt.clf()

# %%
# define the prediction length for testing 
prediciton_length = 70
generated_predicitons = []

# iterate over each batch 
for batch in test_loader:

    past_values, future_values = split_past_future(batch[0], num_future_points=prediciton_length)

    batch = {
            "past_values": past_values,  # (batch_size, input_length, input_size)
            "future_values": future_values,  # (batch_size, prediction_length, input_size)
            "past_time_features": torch.arange(past_values.size(1)).unsqueeze(0).unsqueeze(2).float().repeat(past_values.size(0), 1, 1),  # (batch_size, seq_length, 1)
            "past_observed_mask": torch.ones_like(past_values),  # (batch_size, seq_length, input_size)
            "future_time_features": torch.arange(past_values.size(1), past_values.size(1) + num_future_points).unsqueeze(0).unsqueeze(2).float().repeat(future_values.size(0), 1, 1),  # (batch_size, prediction_length, 1)
        }
    
    # set the model to evaluation mode  
    model.eval()

    # create predicitons 
    with torch.no_grad():
        predictions=model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            future_time_features=batch["future_time_features"], 
            past_observed_mask=batch["past_observed_mask"], 
        )

    generated_batch_predictions = predictions.sequences
    if generated_batch_predictions.shape[0] == 16:
        generated_predicitons.append(generated_batch_predictions)

# %%
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# %%
generated_predicitons = np.array(generated_predicitons)
num_batches = generated_predicitons.shape[0]
batch_size = generated_predicitons.shape[1]
samples = generated_predicitons.shape[2]
seq_length = generated_predicitons.shape[3]
input_size = generated_predicitons.shape[4]

generated_predicitons = generated_predicitons.reshape(num_batches * batch_size, samples, seq_length, input_size)

# %%
sample = 10 
mean_over_obsv = np.mean(generated_predicitons, axis=0)

# %%
pred_mean = np.mean(generated_predicitons, axis=1)
grand_mean = np.mean(mean_over_obsv, axis=0)

# %%
generated_predicitons.shape

# %%
grand_mean.shape

# %%
# define observation you want to examine 
pred = 60
pl = 70
sample = 40
# Plot the best predictions
plt.figure(figsize=(10, 6))

# Plot the average past values
plt.plot(np.arange(len(data_avg[:-pl])), data_avg[:-pl, 0], label="Average of the past sine values")
plt.plot(np.arange(len(data_avg[:-pl])), data_avg[:-pl, 1], label="Average of the past cosine values")

# plot average over observations 
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), mean_over_obsv[sample, :, 0], label="Average of observations", linestyle="dotted")
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), mean_over_obsv[sample, :, 1], label="Average of observations", linestyle="dotted")


# Plot the past values of the observation 
# plt.plot(np.arange(len(data_avg[:-pl])), y[pred, :-pl, 0], label=f"Past sine values")
# plt.plot(np.arange(len(data_avg[:-pl])), y[pred, :-pl, 1], label=f"Past cosine values")

# Plot true future values
plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), data_avg[-pl:, 0], label="Average of future sine values", linestyle='dashed')
plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), data_avg[-pl:, 1], label="Average of future cosine values", linestyle='dashed')

# Plot the grand mean 
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), grand_mean[-pl:, 0], label="Grand mean", linestyle='dashed')
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), grand_mean[-pl:, 1], label="Grand mean", linestyle='dashed')

# Plot predicted future values of one sampled observation 
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), generated_predicitons[pred, sample, :, 0], label=f"Predicted Sine Values of {pred + 1}th observation", linestyle='dotted')
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), generated_predicitons[pred, sample, :, 1], label=f"Predicted Cosine Values of {pred + 1}th observation", linestyle='dotted')

# Plot the mean prediction 
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), pred_mean[pred, :, 0], label="Average of samples for sine")
# plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), pred_mean[pred, :, 1], label="Average of samples for cosine")

# Plot the mean over epochs 
plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), sine_mu_across_samples, label="Mean over epochs", linestyle="dotted")
plt.plot(np.arange(len(data_avg[:-pl]), len(data_avg[:-pl]) + pl), cosine_mu_across_samples, label="Mean over epochs", linestyle="dotted")


plt.legend(loc='upper right')
plt.savefig("Predicitons.png")
plt.clf()


# %%



