import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AIContentDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("devloverumar/chatgpt-content-detector")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForSequenceClassification.from_pretrained("devloverumar/chatgpt-content-detector")
        self.max_len = self.model.config.max_position_embeddings

    def rate_content(self, content):
        inputs = self.tokenizer(content, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)

        ai_probability = predictions[0][1].item()
        ai_probability = np.clip(ai_probability, 0.0, 1.0)

        if ai_probability > 0.8:
            reward = -1  # High penalty for high AI probability
        else:
            reward = 1.0 - ai_probability  # Reward decreases as AI probability increases

        return reward
