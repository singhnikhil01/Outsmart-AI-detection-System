import torch
import numpy as np
from nltk.tokenize import sent_tokenize

class RewardCalculator:
    def __init__(self, aicd_model, grammar_checker=None, fact_checker=None):
        self.aicd_model = aicd_model
        self.grammar_checker = grammar_checker
        self.fact_checker = fact_checker

    def get_reward(self, texts_array):
        rewards = []
        for text in texts_array:
            result = self.calculate_reward(text)
            rewards.append(torch.tensor(result))
        return rewards

    def calculate_reward(self, text):
        aicd_score = self.aicd_model.rate_content(text)
        grammar_penalty = 0
        if self.grammar_checker:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                try:
                    gram = self.grammar_checker.check(sentence)
                    if gram:
                        grammar_penalty -= 0.5
                except:
                    grammar_penalty -= 0.5
        reward = aicd_score + grammar_penalty
        return reward
