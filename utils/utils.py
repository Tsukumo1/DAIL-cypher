import collections
import json
import os
import re


from transformers import AutoTokenizer
from utils.enums import LLM


def filter_json(raw_response: str) -> str:
    try:
        id_s = raw_response.index("{")
        id_e = raw_response.rindex("}")
        if id_s > id_e:
            raise ValueError("Wrong json format")
        else:
            return raw_response[id_s: id_e + 1]
    except ValueError:
        raise ValueError("Wrong json format")


def cost_estimate(n_tokens: int, model):
    return LLM.costs_per_thousand[model] * n_tokens / 1000


def get_neo4j_for_database(input):
    with open('./neo4j/schemas.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        if item.get('db_id') == input:
            neo4j = simplify_cypher(item) # simplification
            # neo4j = item['orig']
    return neo4j

def simplify_cypher(schema_data):
    nodes = schema_data.get('Node_original', [])
    node_properties = schema_data.get('Node_properties_original', [])
    edges = schema_data.get('Edge_original', [])
    edge_properties = schema_data.get('Edge_properties_original', [])
    orig_text = schema_data.get('orig', '')

    results = []
    results.append('Nodes:')

    for i, node in enumerate(nodes):
        node_info = f"{node}"
        node_props = [prop[1] for prop in node_properties if prop[0] == i]
        if node_props:
            node_info += f" | Properties: {node_props}"
        results.append(node_info)


    results.append('\nEdges:')
    for i, edge in enumerate(edges):
        edge_info = f"{edge}"
        edge_props = [prop[1] for prop in edge_properties if prop[0] == i]
        if edge_props:
            edge_info += f" | Properties: {edge_props}"
        results.append(edge_info)


    relationships_text = ""
    if "The relationships:" in orig_text:
        relationships_text = orig_text.split("The relationships:")[-1].strip()
    

    final_result = "\n".join(results)
    if relationships_text:
        final_result += f"\n\nRelationships Information:\n{relationships_text}"
    return final_result



def tokenize_cypher(cypher):
    tokens = []
    current_token = ''
    in_string = False
    for char in cypher:
        if char in [' ', '\n', '\t'] and not in_string:
            if current_token:
                tokens.append(current_token)
                current_token = ''
        elif char == "'" and not in_string:
            in_string = True
            current_token += char
        elif char == "'" and in_string:
            in_string = False
            current_token += char
            tokens.append(current_token)
            current_token = ''
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def cypher2skeleton(cypher: str, graph_schema):
    cypher = cypher_normalization(cypher)

    relationship_name = [label.lower() for label in graph_schema["Edge_original"]]
    relationship_properties = [label[1].lower() for label in graph_schema["Edge_properties_original"]]
    node_name = [rel.lower() for rel in graph_schema["Node_original"]]
    node_properties = [rel[1].lower() for rel in graph_schema["Node_properties_original"]]

    tokens = tokenize_cypher(cypher)
    new_cypher_tokens = []
    for token in tokens:
        token_lower = token.lower()
        # mask node labels
        if token_lower in node_name or token_lower in node_properties \
                or token_lower in relationship_name or token_lower in relationship_properties:
            new_cypher_tokens.append("_")
        # mask string literals
        elif token.startswith("'") and token.endswith("'"):
            new_cypher_tokens.append("_")
        # mask number literals
        elif token.replace('.', '').isdigit():
            new_cypher_tokens.append("_")
        else:
            new_cypher_tokens.append(token_lower)

    cypher_skeleton = " ".join(new_cypher_tokens)

    # remove pattern matching details
    cypher_skeleton = re.sub(r'\([\w:_\s]+\)', '(_)', cypher_skeleton)
    cypher_skeleton = re.sub(r'\[[\w:_\s]+\]', '[_]', cypher_skeleton)

    # simplify WHERE clauses
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in cypher_skeleton:
            cypher_skeleton = cypher_skeleton.replace("_ {} _".format(op), "_")
    while ("where _ and _" in cypher_skeleton or "where _ or _" in cypher_skeleton):
        if "where _ and _" in cypher_skeleton:
            cypher_skeleton = cypher_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in cypher_skeleton:
            cypher_skeleton = cypher_skeleton.replace("where _ or _", "where _")

    # simplify WITH and RETURN clauses
    cypher_skeleton = re.sub(r'with[\s\w,_]+', 'with _', cypher_skeleton, flags=re.IGNORECASE)
    cypher_skeleton = re.sub(r'return[\s\w,_]+', 'return _', cypher_skeleton, flags=re.IGNORECASE)

    # remove additional spaces in the skeleton
    cypher_skeleton = re.sub(r'\s+', ' ', cypher_skeleton).strip()

    return cypher_skeleton

def cypher_normalization(cypher):
    cypher = cypher.strip()

    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                in_quotation = not in_quotation

        return out_s

    def remove_semicolon(s):
        return s[:-1] if s.endswith(";") else s

    def double2single(s):
        return s.replace('"', "'")

    def add_asc(s):
        pattern = re.compile(r'order by (?:\w+\(?\s*\S+\s*\)?|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+\(?\s*\S+\s*\)?|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")
        return s

    def remove_variable_aliases(s):
        tokens = tokenize_cypher(s)
        aliases = {}
        new_s = []

        for i, token in enumerate(tokens):
            if token.lower() == 'as':
                if i > 0 and i < len(tokens) - 1:
                    aliases[tokens[i+1]] = tokens[i-1]
            elif token in aliases:
                new_s.append(aliases[token])
            else:
                new_s.append(token)

        return ' '.join(new_s)

    processing_func = lambda x: remove_variable_aliases(add_asc(lower(double2single(remove_semicolon(x)))))

    return processing_func(cypher)

def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True


def jaccard_similarity(skeleton1, skeleton2):
    tokens1 = skeleton1.strip().split(" ")
    tokens2 = skeleton2.strip().split(" ")
    total = len(tokens1) + len(tokens2)

    def list_to_dict(tokens):
        token_dict = collections.defaultdict(int)
        for t in tokens:
            token_dict[t] += 1
        return token_dict
    token_dict1 = list_to_dict(tokens1)
    token_dict2 = list_to_dict(tokens2)

    intersection = 0
    for t in token_dict1:
        if t in token_dict2:
            intersection += min(token_dict1[t], token_dict2[t])
    union = (len(tokens1) + len(tokens2)) - intersection
    return float(intersection) / union
