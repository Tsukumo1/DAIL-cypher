import json.decoder
import openai
from utils.enums import LLM
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import dashscope
from dashscope import Generation
import time

def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    if model == LLM.TONG_YI_QIAN_WEN or model == LLM.LLAMA_3_8B or model == LLM.LLAMA_2_13B or model == LLM.QWEN_PLUS:
        dashscope.api_key = OPENAI_API_KEY
    else:
        openai.api_key = OPENAI_API_KEY
        openai.organization = OPENAI_GROUP_ID

def ask_completion(model, batch, temperature):
    response = openai.Completion.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )

def ask_chat(model, messages: list, temperature, n):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=200,
        n=n
    )
    response_clean = [choice["message"]["content"] for choice in response["choices"]]
    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        **response["usage"]
    )
    
def ask_llm(model_name, model, batch: list, temperature: float, n: int):
    n_repeat = 0
    while True:
        try:
            # print(model)
            # print(LLM.QWEN_PLUS)
            messages = [{"role": "system", "content": '''
Given an input question, convert it to a Cypher query.
To translate a question into a Cypher query, please follow these steps:

1. Carefully analyze the provided graph schema to understand what nodes, relationships, and properties are available. Pay attention to the node labels, relationship types, and property keys.

2. Identify the key entities and relationships mentioned in the natural language question. Map these to the corresponding node labels, relationship types, and properties in the graph schema.

3. Think through how to construct a Cypher query to retrieve the requested information step-by-step. Focus on:
   - Identifying the starting node(s) 
   - Traversing the necessary relationships
   - Filtering based on property values
   - Returning the requested information
Feel free to use multiple MATCH, WHERE, and RETURN clauses as needed.

4. Once you have finished constructing the Cypher query, provide the final query inside triple backticks ```cypher```.

5. Explain how your Cypher query will retrieve the necessary information from the graph to answer the original question. Provide this explanation inside <explanation> tags.

Remember, the goal is to construct a Cypher query that will retrieve the relevant information to answer the question based on the given graph schema. Carefully map the entities and relationships in the question to the nodes, relationships, and properties in the schema.

If the question cannot be answered by the information in the graph schema, say "Based on the given graph schema, there is not enough information to answer this question." inside ```cypher``` tags. Do not attempt to construct a query if the schema does not support it.

                             '''},
                        {"role": "user", "content": batch[0]}]
            if model_name == LLM.META_LLAMA_3_8B:
                len(batch) == 1
                messages = [{"role": "user", "content": batch[0]}]
                response = model.ask_llama(messages, temperature)
            elif model_name == LLM.QWEN_2_7B:
                len(batch) == 1
                messages = [{"role": "user", "content": batch[0]}]
                response = model.ask_qwen(messages, temperature)
            elif model_name == LLM.DEEP_SEEK:
                len(batch) == 1
                response = model.ask_deepseek(messages, temperature)
            break
        except openai.error.RateLimitError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for RateLimitError", end="\n")
            time.sleep(1)
            continue
        except json.decoder.JSONDecodeError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
            time.sleep(1)
            continue
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1)
            continue

    return response


