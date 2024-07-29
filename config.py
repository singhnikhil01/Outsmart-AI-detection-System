from trl import PPOConfig
import torch

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
