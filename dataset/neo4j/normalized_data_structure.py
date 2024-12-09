import pandas as pd
df = pd.read_csv('orig_data.csv')

import nltk
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')
def token_process(input_string, if_replace=False):
    tokens = word_tokenize(input_string)
    replaced_tokens = None
    if if_replace:
        replaced_tokens = [re.sub(r'\d+', 'value', token) for token in tokens]
    return [tokens, replaced_tokens]


data_list = [
    {'db_id': row['database'], 'query': row['cypher'], 'question': row['question'], 
     'query_toks': token_process(row['cypher'], True)[0], 
     "query_toks_no_value": token_process(row['cypher'], True)[1],
     'question_toks': token_process(row['question'])[0]}
    for _, row in df.iterrows()
]

import random


data_index=list(range(len(data_list)))
random.shuffle(data_index)


split_index = int(len(data_list) * 0.8)


train_index = data_index[:split_index]
test_index = data_index[split_index:]

train_data = [data_list[i] for i in train_index if i < len(data_list)]
test_data = [data_list[i] for i in test_index if i < len(data_list)]
print("Training Data:", len(train_data))
print("Testing Data:", len(test_data))

import json

with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("already processed train and test data.")


## schemas
df_unique = df[['database', 'schema']].drop_duplicates()

def extract_properties(input_string):
    result = {}

    # 1. 提取 Node_original
    node_properties_section = re.search(r'Node properties:(.*)Relationship properties:', input_string, re.DOTALL)
    if node_properties_section:
        node_properties_text = node_properties_section.group(1)
        result['Node_original'] = re.findall(r'\*\*(.*?)\*\*', node_properties_text)

    # 2. 提取 Node_properties_original
    node_splits = re.split(r'\*\*.*?\*\*', node_properties_text)
    node_properties_original = []
    
    for i in range(1, len(node_splits)):
        lines = re.split(r'\n', node_splits[i])
        for line in lines:
            # Extract the first backtick content in each line
            match = re.search(r'`(.*?)`', line)
            if match:
                prop = match.group(1)
                node_properties_original.append([i-1, prop])
    
    result['Node_properties_original'] = node_properties_original

    # 3. 提取 Node_properties_type (更新这部分)
    node_properties_type = []
    lines = re.split(r'\n', node_properties_text)
    for line in lines:
        colon_count = line.count(':')
        if colon_count >= 2:
            segments = re.split(r':\s*', line)
            if len(segments) >= 3:
                text_between_colons = segments[1]
                uppercase_words = re.findall(r'\b[A-Z_]+\b', text_between_colons)
                uppercase_words = [word.replace('_', ' ') for word in uppercase_words]
                node_properties_type.extend(uppercase_words)
        elif colon_count == 1:
            match = re.search(r': (.*?)$', line)
            if match:
                line_text = match.group(1)
                uppercase_words = re.findall(r'\b[A-Z_]+\b', line_text)
                uppercase_words = [word.replace('_', ' ') for word in uppercase_words]
                node_properties_type.extend(uppercase_words)
    
    result['Node_properties_type'] = node_properties_type

    # 4. 提取 relationships
    relationships_section = re.search(r'The relationships:(.*)', input_string, re.DOTALL)
    if relationships_section:
        relationships_text = relationships_section.group(1)
        relationships = re.findall(r'\(\s*:(.*?)\s*\)\s*-\s*\[:(.*?)\]\s*->\s*\(\s*:(.*?)\s*\)', relationships_text)
        if relationships:
            reshape_relation = []
            for r in relationships:
                reshape_relation.append([r[0],r[1],r[2]])
            result['relationship']=reshape_relation

    # 5. 新增：处理 Node
    result['Node'] = []
    for node in result['Node_original']:
        processed_node = re.findall(r'[A-Z][a-z]*', node)
        processed_node = ' '.join(processed_node).lower()
        result['Node'].append(processed_node)

    # 6. 新增：处理 Node_properties
    result['Node_properties'] = []
    for _, prop in result['Node_properties_original']:
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', prop)
        
        processed_words = []
        for i, word in enumerate(words):
            if word.isupper() and len(word) > 1:
                if i < len(words) - 1 and not words[i+1].isupper():
                    processed_words.append(word[:-1].lower())
                    words[i+1] = word[-1] + words[i+1]
                else:
                    processed_words.append(word.lower())
            else:
                processed_words.append(word.lower())
        
        processed_prop = ' '.join(processed_words)
        processed_prop = processed_prop.replace('_', ' ')
        result['Node_properties'].append(processed_prop)
        
    return result


schemas = []
for index, row in df_unique.iterrows():
    result = extract_properties(row[1])
    result['db_id']=row[0]
    schemas.append(result)

with open('schemas.json', 'w', encoding='utf-8') as f:
    json.dump(schemas, f, ensure_ascii=False, indent=4)


print("already processed schemas.")
