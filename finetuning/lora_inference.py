import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_cypher_model_ds2"
base_model_path = "path-to-base-model"  # Ensure this is the correct base model path
base_model = AutoModelForCausalLM.from_pretrained(base_model_path,trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True, torch_dtype=torch.bfloat16)

peft_config = PeftConfig.from_pretrained(model_path)
model = PeftModel.from_pretrained(base_model, model_path)
model.config.pad_token_id = model.config.eos_token_id

model.to(device)
model.eval()

with open('test_questions_DAIL_3shot.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
questions = data['questions']

system_prompt = """
Given an input question, convert it to a Cypher query.
Do not given any explain. Return a clear cypher query.
"""

# Inference function
def generate_response(question):
    input_text = f"System:{system_prompt}\n\nUser:{question[:-8]}\n\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=input_length + 512, num_return_sequences=1, temperature=0.1)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Process the response: replace newlines with spaces and strip
    processed_response = ' '.join(response.split())
    return processed_response

# Perform inference and save results
print("Starting Cypher query generation...")
output_file = 'results_lora_dscoder2.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for question_data in tqdm(questions, desc="Processing questions"):
        question = question_data['prompt']
        cypher_query = generate_response(question)
        f.write(f"{cypher_query}\n")
