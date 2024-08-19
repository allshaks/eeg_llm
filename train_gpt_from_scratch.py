from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import numpy as np 
from torch.optim import Adam 
import matplotlib.pyplot as plt 
import torch

# generate noisy sine and cosine waves as toy data
def generate_wave_data(num_samples=1000, seq_length=100):
    t = np.linspace(0, 4*np.pi, seq_length)
    sine = np.sin(t) + 0.1 * np.random.randn(seq_length)
    cosine = np.cos(t) + 0.1 * np.random.randn(seq_length)

    data = np.stack([sine, cosine], axis=1)


    return data

data = generate_wave_data()
# divide in past and future values for training the transformer 
past_values = torch.tensor(data[:-10], dtype=torch.float32)
future_values = torch.tensor(data[-10:], dtype=torch.float32)

# define the configuration of the model 
config = TimeSeriesTransformerConfig(
    prediction_length=10,  # length of the future values to predict
    context_length=89,  # length of the past values to use
    num_time_features=1,
    #num_dynamic_real_features=2,  # the number of time features (here we have only time steps)
    encoder_layers=2,  # number of transformer layers in the encoder
    decoder_layers=2,  # number of transformer layers in the decoder
    d_model=32,  # dimension of the model
    n_heads=4,  # number of attention heads
    input_size=2,
    lags_sequence=[1], 
    )

# initialize the model 
model = TimeSeriesTransformerForPrediction(config)

# Modified batch with correct dimensions
batch = {
    "past_values": past_values.unsqueeze(0),  # (1, seq_length, 2)
    "future_values": future_values.unsqueeze(0),  # (1, prediction_length, 2)
    "past_time_features": torch.arange(len(past_values)).unsqueeze(0).unsqueeze(2).float(),  # (1, seq_length, 1)
    "past_observed_mask": torch.ones_like(past_values).unsqueeze(0),  # (1, seq_length, 2)
    "future_time_features": torch.arange(len(past_values), len(past_values) + 10).unsqueeze(0).unsqueeze(2).float(),  # (1, prediction_length, 1)
}

# Initialize optimizer
optim = Adam(model.parameters(), lr=1e-3)

# Define training loop 
num_epochs = 100 

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
    )

    loss = outputs.loss

    # Backward pass
    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

""" from huggingface_hub import hf_hub_download
import torch
from transformers import TimeSeriesTransformerModel

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

# during training, one provides both past and future values
# as well as possible additional features
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
)

last_hidden_state = outputs.last_hidden_state """