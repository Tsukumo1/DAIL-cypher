import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import os
from transformers.trainer_callback import TrainerCallback

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

model_address = "path-to-model"

tokenizer = AutoTokenizer.from_pretrained(model_address, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_address, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 获取 questions.json 文件路径
questions_file = 'train_questions_DAIL_3shot.json'

with open(questions_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

if 'questions' not in data:
    raise ValueError("'questions' don't exist.")


def filter_questions_by_token_length(questions, tokenizer, max_token_length=1200):  

    filtered_questions = []
    for question in questions:
        prompt = question['prompt']
        if prompt:

            encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            token_length = len(encoded_prompt)

            if token_length <= max_token_length:
                filtered_questions.append(question)
    
    
    return filtered_questions

questions = filter_questions_by_token_length(data['questions'], tokenizer)
total_questions = len(questions)

system_prompt = """
Given an input question, convert it to a Cypher query.
"""

class CypherDataset(Dataset):
    def __init__(self, questions, tokenizer, max_length=1200):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_data = self.questions[idx]
        prompt = question_data['prompt']


        # instruction = self.tokenizer(f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt[:-8]}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        instruction = self.tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt[:-8]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        response = self.tokenizer('MATCH ('+ question_data['response'], add_special_tokens=False)  # 假设'target'字段包含目标Cypher查询
        input_length = len(instruction)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]  
        if len(input_ids) > self.max_length:  # 做一个截断
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            print('cut')
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


dataset = CypherDataset(questions, tokenizer)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules = ["q_proj", "v_proj"],
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=16,
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class PrinterCallback(TrainerCallback):
    def __init__(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        print(f"\nEpoch {self.current_epoch}/{self.num_epochs}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 == 0:
            loss = logs.get('loss', 'N/A')
            print(f"Step: {state.global_step}/{self.num_batches * self.num_epochs}, Loss: {loss}")


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[PrinterCallback(num_epochs=training_args.num_train_epochs, 
                               num_batches=len(train_loader))]
)

# 开始训练
print("Finetuning begin...")
trainer.train()

# 保存微调后的模型
print("Save the fine-tuned model...")
model.save_pretrained("./fine_tuned_cypher_model_ds2")
tokenizer.save_pretrained("./fine_tuned_cypher_model_ds2")
