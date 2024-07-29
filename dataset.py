from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler
from config import config

def build_dataset(dataset_name="humarin/chatgpt-paraphrases", input_min_text_length=2, input_max_text_length=512):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name)['train']
    ds = ds.rename_column("text", "prompt")
    ds = ds.remove_columns(['paraphrases', 'category', 'source'])
    ds = ds.filter(lambda x: len(x["prompt"]) > 2, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        input_ids = tokenizer.encode(sample["prompt"], truncation=True, max_length=256)
        query = tokenizer.decode(input_ids)
        return {"input_ids": input_ids, "query": query}

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
