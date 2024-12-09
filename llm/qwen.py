import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("path-to/qwen2.5_coder", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("path-to/qwen2.5_coder", trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def ask_qwen(self, messages, temperature=0, max_tokens=200):
        # 验证输入格式
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
            raise ValueError("messages must be a list of dictionaries, each containing 'role' and 'content' keys")

        # 将消息列表转换为单个提示字符串
        prompt = self.format_messages(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_response = full_response[len(prompt):].strip()
    
        return {
            'response': new_response,
            'prompt_tokens': len(self.tokenizer.encode(prompt)),
            'completion_tokens': len(self.tokenizer.encode(full_response)),
            'total_tokens': len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(full_response))
        }

    def format_messages(self, messages):
        formatted_prompt = ""
        for message in messages:
            if message['role'] == 'user':
                formatted_prompt += f"Human: {message['content']}\n"
            elif message['role'] == 'assistant':
                formatted_prompt += f"Assistant: {message['content']}\n"
            # 可以根据需要添加其他角色
        formatted_prompt += "Assistant: "
        return formatted_prompt

qwen = Qwen()