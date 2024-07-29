import torch
from transformers import AutoTokenizer
from trl import PPOTrainer
from tqdm import tqdm
import nltk
import language_tool_python

from config import config, device
from model import model, tokenizer
from dataset import build_dataset
from detector import AIContentDetector
from reward_calculator import RewardCalculator

nltk.download('punkt')
tool = language_tool_python.LanguageTool('en-US')

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataset = build_dataset()
ppo_trainer = PPOTrainer(
    config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 250,  # Limit the number of tokens generated as output
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

detector = AIContentDetector()
reward_calculator = RewardCalculator(detector, tool)

epochs = 20

for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        rewards = reward_calculator.get_reward(response_texts)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        ppo_trainer.save_pretrained("fine_tuned_gpt2")
        tokenizer.save_pretrained("fine_tuned_gpt2")