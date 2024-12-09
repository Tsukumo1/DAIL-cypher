import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_address = "path-to-model"

tokenizer = AutoTokenizer.from_pretrained(model_address, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_address, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

questions_file = 'questions_basic.json'
results_file = 'result_dscode2_basic.txt'  # 保存为txt文件


with open(questions_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


if 'questions' not in data:
    raise ValueError("'questions' don't exist.")

questions = data['questions']
total_questions = len(questions)

system = """
Given an input question, convert it to a Cypher query.
Once you have finished constructing the Cypher query, provide the final query inside triple backticks ```cypher```.
Remember, the goal is to construct a Cypher query that will retrieve the relevant information to answer the question based on the given graph schema. Carefully map the entities and relationships in the question to the nodes, relationships, and properties in the schema.
"""


with open(results_file, 'w', encoding='utf-8') as result_file:

    for question_data in tqdm(questions, desc="处理问题", unit="问题"):
        if 'prompt' not in question_data:
            continue
        prompt = question_data['prompt']
        input_text = f"System:{system} \n\n User:{prompt[:-8]}\n\nAssistant:"
        

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_reply = decoded_output.split("Assistant:")[-1].strip()
        assistant_reply_single_line = assistant_reply.replace('\n', ' ').strip()
        
        result_file.write(f"{assistant_reply_single_line}\n")