import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekCoderV2:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def init_model(self):
        # 使用指定路径加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained("path-to/deepseek_code_v2", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("path-to/deepseek_code_v2", trust_remote_code=True, torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def ask_deepseek(self, messages, temperature=0, max_tokens=200):
        # 验证messages的格式
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
            raise ValueError("messages 必须是一个包含字典的列表，每个字典需包含 'role' 和 'content' 键")
        
        # 将消息列表格式化为提示字符串
        prompt = self.format_messages(messages)

        # 编码输入并移动到 GPU（如果可用）
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # 生成响应并解码
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                do_sample=True
            )

        # 处理并返回响应
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_response = full_response[len(prompt):].strip()

        return {
            'response': new_response,
            'prompt_tokens': len(self.tokenizer.encode(prompt)),
            'completion_tokens': len(self.tokenizer.encode(full_response)),
            'total_tokens': len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(full_response))
        }

    def format_messages(self, messages):
        # 将消息格式化为单个提示字符串
        formatted_prompt = ""
        for message in messages:
            if message['role'] == 'user':
                formatted_prompt += f"Human: {message['content']}\n"
            elif message['role'] == 'assistant':
                formatted_prompt += f"Assistant: {message['content']}\n"
        formatted_prompt += "Assistant: "
        return formatted_prompt

# 实例化并初始化模型
deepseek_coder_v2 = DeepSeekCoderV2()