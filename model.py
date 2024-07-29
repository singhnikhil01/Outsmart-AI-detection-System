from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from config import config, device

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
