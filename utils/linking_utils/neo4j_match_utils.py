import re
import string
import collections

import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)

CELL_EXACT_MATCH_FLAG = "EXACTMATCH"
CELL_PARTIAL_MATCH_FLAG = "PARTIALMATCH"
COL_PARTIAL_MATCH_FLAG = "CPM"
COL_EXACT_MATCH_FLAG = "CEM"
TAB_PARTIAL_MATCH_FLAG = "TPM"
TAB_EXACT_MATCH_FLAG = "TEM"

# schema linking, similar to IRNet
def compute_schema_linking(question, column, table, type):
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = COL_EXACT_MATCH_FLAG
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = TAB_EXACT_MATCH_FLAG

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = COL_PARTIAL_MATCH_FLAG
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = TAB_PARTIAL_MATCH_FLAG
        n -= 1
        if type == 'node':
            return {"q_np_match": q_col_match, "q_node_match": q_tab_match}
        elif type == 'edge':
            return {"q_ep_match": q_col_match, "q_edge_match": q_tab_match}



def match_shift(q_col_match, q_tab_match):

    q_id_to_match = collections.defaultdict(list)
    for match_key in q_col_match.keys():
        q_id = int(match_key.split(',')[0])
        c_id = int(match_key.split(',')[1])
        type = q_col_match[match_key]
        q_id_to_match[q_id].append((type, c_id))
    for match_key in q_tab_match.keys():
        q_id = int(match_key.split(',')[0])
        t_id = int(match_key.split(',')[1])
        type = q_tab_match[match_key]
        q_id_to_match[q_id].append((type, t_id))
    relevant_q_ids = list(q_id_to_match.keys())

    priority = []
    for q_id in q_id_to_match.keys():
        q_id_to_match[q_id] = list(set(q_id_to_match[q_id]))
        priority.append((len(q_id_to_match[q_id]), q_id))
    priority.sort()
    matches = []
    new_q_col_match, new_q_tab_match = dict(), dict()
    for _, q_id in priority:
        if not list(set(matches) & set(q_id_to_match[q_id])):
            exact_matches = []
            for match in q_id_to_match[q_id]:
                if match[0] in [COL_EXACT_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                    exact_matches.append(match)
            if exact_matches:
                res = exact_matches
            else:
                res = q_id_to_match[q_id]
            matches.extend(res)
        else:
            res = list(set(matches) & set(q_id_to_match[q_id]))
        for match in res:
            type, c_t_id = match
            if type in [COL_PARTIAL_MATCH_FLAG, COL_EXACT_MATCH_FLAG]:
                new_q_col_match[f'{q_id},{c_t_id}'] = type
            if type in [TAB_PARTIAL_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                new_q_tab_match[f'{q_id},{c_t_id}'] = type

    return new_q_col_match, new_q_tab_match