from utils.linking_utils.neo4j_match_utils import match_shift

def mask_question_with_schema_linking(data_jsons, mask_tag, value_tag):
    mask_questions = []
    for data_json in data_jsons:
        sc_link_node = data_json["sc_link_node"]
        sc_link_edge = data_json["sc_link_edge"]
        # cv_link = data_json["cv_link"]
        q_np_match = sc_link_node["q_np_match"]
        q_node_match = sc_link_node["q_node_match"]
        q_ep_match = sc_link_edge["q_ep_match"]
        q_edge_match = sc_link_edge["q_edge_match"]
        # num_date_match = cv_link["num_date_match"]
        # cell_match = cv_link["cell_match"]
        question_for_copying = data_json["question_for_copying"]

        q_np_match, q_node_match = match_shift(q_np_match, q_node_match)
        q_ep_match, q_edge_match = match_shift(q_ep_match, q_edge_match)

        def mask(question_toks, mask_ids, tag):
            new_question_toks = []
            for id, tok in enumerate(question_toks):
                if id in mask_ids:
                    new_question_toks.append(tag)
                else:
                    new_question_toks.append(tok)
            return new_question_toks

        # num_date_match_ids = [int(match.split(',')[0]) for match in num_date_match]
        # cell_match_ids = [int(match.split(',')[0]) for match in cell_match]
        # value_match_q_ids = num_date_match_ids + cell_match_ids
        # question_toks = mask(question_for_copying, value_match_q_ids, value_tag)

        q_np_match_ids = [int(match.split(',')[0]) for match in q_np_match]
        q_node_match_ids = [int(match.split(',')[0]) for match in q_node_match]
        q_ep_match_ids = [int(match.split(',')[0]) for match in q_ep_match]
        q_edge_match_ids = [int(match.split(',')[0]) for match in q_edge_match]
        schema_match_q_ids = q_np_match_ids + q_node_match_ids + q_ep_match_ids + q_edge_match_ids
        question_toks = mask(question_for_copying, schema_match_q_ids, mask_tag)
        mask_questions.append(" ".join(question_toks))

    return mask_questions


def get_question_pattern_with_schema_linking(data_jsons):
    question_patterns = []
    for data_json in data_jsons:
        sc_link_node = data_json["sc_link_node"]
        sc_link_edge = data_json["sc_link_edge"]
        # cv_link = data_json["cv_link"]
        q_np_match = sc_link_node["q_np_match"]
        q_node_match = sc_link_node["q_node_match"]
        q_ep_match = sc_link_edge["q_ep_match"]
        q_edge_match = sc_link_edge["q_edge_match"]
        # num_date_match = cv_link["num_date_match"]
        # cell_match = cv_link["cell_match"]
        question_for_copying = data_json["question_for_copying"]

        def mask(question_toks, mask_ids, tag):
            new_question_toks = []
            for id, tok in enumerate(question_toks):
                if id in mask_ids:
                    new_question_toks.append(tag)
                else:
                    new_question_toks.append(tok)
            return new_question_toks


        q_np_match_ids = [int(match.split(',')[0]) for match in q_np_match]
        q_node_match_ids = [int(match.split(',')[0]) for match in q_node_match]
        q_ep_match_ids = [int(match.split(',')[0]) for match in q_ep_match]
        q_edge_match_ids = [int(match.split(',')[0]) for match in q_edge_match]
        schema_match_q_ids = q_np_match_ids + q_node_match_ids + q_ep_match_ids + q_edge_match_ids
        question_toks = mask(question_for_copying, schema_match_q_ids, '_')
        question_patterns.append(" ".join(question_toks))

    return question_patterns


def get_relevant_tables(data_jsons, RELEVANT_TABLE_BADCASE, RELEVANT_TABLE_TOTALCASE):
    relevant_tables = []
    for data_json in data_jsons:
        table_names = data_json['table_names_original']
        col_to_tab = data_json['column_to_table']
        q_col_match = data_json['sc_link']['q_col_match']
        q_tab_match = data_json['sc_link']['q_tab_match']
        cell_match = data_json['cv_link']['cell_match']

        relevant_table_ids = []

        #### all relevant tables ####
        for match_key in q_col_match.keys():
            q_id = int(match_key.split(',')[0])
            t_id = col_to_tab[match_key.split(',')[1]]
            relevant_table_ids.append(t_id)
        for match_key in q_tab_match.keys():
            q_id = int(match_key.split(',')[0])
            t_id = int(match_key.split(',')[1])
            relevant_table_ids.append(t_id)
        for match_key in cell_match.keys():
            if cell_match[match_key] == "EXACTMATCH":
                q_id = int(match_key.split(',')[0])
                t_id = col_to_tab[match_key.split(',')[1]]
                relevant_table_ids.append(t_id)

        relevant_table_ids = list(set(relevant_table_ids))

        relevant_table_names = [table_names[id] for id in relevant_table_ids]
        if len(relevant_table_names) == 0:
            relevant_table_names = table_names

        relevant_tables.append(relevant_table_names)

        RELEVANT_TABLE_TOTALCASE = RELEVANT_TABLE_TOTALCASE + 1
        true_relevant_table_names = []
        query = data_json["query"].lower()
        for token in query.split():
            for table_name in table_names:
                if table_name.lower() in token.split('.'):
                    true_relevant_table_names.append(table_name)
        true_relevant_table_names = list(set(true_relevant_table_names))

        for true_table in true_relevant_table_names:
            if true_table not in relevant_table_names:
                RELEVANT_TABLE_BADCASE = RELEVANT_TABLE_BADCASE + 1
                break

    return relevant_tables, RELEVANT_TABLE_BADCASE, RELEVANT_TABLE_TOTALCASE

