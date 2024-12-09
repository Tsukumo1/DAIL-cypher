import collections
import itertools
import json
import os

import attr
import numpy as np
import torch

from utils.linking_utils import abstract_preproc, corenlp, serialization
from utils.linking_utils.neo4j_match_utils import (
    compute_schema_linking
)

@attr.s
class PreprocessedSchema:
    np_names = attr.ib(factory=list)
    node_names = attr.ib(factory=list)
    node_bounds = attr.ib(factory=list)
    np_to_node = attr.ib(factory=dict)
    node_to_nps = attr.ib(factory=dict)
    node_point_to = attr.ib(factory=dict)
    node_be_point_to = attr.ib(factory=dict)
    normalized_np_names = attr.ib(factory=list)
    normalized_node_names = attr.ib(factory=list)

    ep_names = attr.ib(factory=list)
    edge_names = attr.ib(factory=list)
    edge_bounds = attr.ib(factory=list)
    ep_to_edge = attr.ib(factory=dict)
    edge_to_eps = attr.ib(factory=dict)
    normalized_ep_names = attr.ib(factory=list)
    normalized_edge_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,
                               bert=False):
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    last_node_id = None
    last_edge_id = None

    for i, np in enumerate(schema.node_properties):
        np_toks = tokenize_func(
            np.name, np.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {np.type}>'
        if bert:
            # for bert, we take the representation of the first word
            np_name = np_toks + [type_tok]
            r.normalized_np_names.append(Bertokens(np_toks))
        else:
            np_name = [type_tok] + np_toks

        
        # if np.node is None:
        #     node_name = ['<any-node>']
        # else:
        #     node_name = tokenize_func(
        #         np.node.name, np.node.unsplit_name)
        # np_name += ['<node-sep>'] + node_name
        r.np_names.append(np_name)

        node_id = None if np.node is None else np.node.id
        r.np_to_node[str(i)] = node_id
        if node_id is not None:
            nps = r.node_to_nps.setdefault(str(node_id), [])
            nps.append(i)
        if last_node_id != node_id:
            r.node_bounds.append(i)
            last_node_id = node_id

        # if column.foreign_key_for is not None:
        #     r.foreign_keys[str(column.id)] = column.foreign_key_for.id
        #     r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.node_bounds.append(len(schema.node_properties))
    assert len(r.node_bounds) == len(schema.nodes) + 1

    for i, node in enumerate(schema.nodes):
        node_toks = tokenize_func(
            node.name, node.unsplit_name)
        r.node_names.append(node_toks)
        if bert:
            r.normalized_node_names.append(Bertokens(node_toks))

        if node.point_to is not None:
            r.node_point_to[str(node.id)]={}
            for i, (target_id, relation) in enumerate(node.point_to):
                relation_toks = tokenize_func(
                    None, relation)
                r.node_point_to[str(node.id)][target_id]=relation

        if node.be_point_to is not None:
            r.node_be_point_to[str(node.id)]={}
            for i, (source_id, relation) in enumerate(node.be_point_to):
                relation_toks = tokenize_func(
                    None, relation)
                r.node_be_point_to[str(node.id)][source_id]=relation
            # print(node.link_to)
    
    # last_table = schema.nodes[-1]

    # r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    # r.primary_keys = [
    #     column.id
    #     for table in schema.tables
    #     for column in table.primary_keys
    # ] if fix_issue_16_primary_keys else [
    #     column.id
    #     for column in last_table.primary_keys
    #     for table in schema.tables
    # ]

    ### Edge
    for i, ep in enumerate(schema.edge_properties):
        ep_toks = tokenize_func(
            ep.name, ep.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {ep.type}>'
        if bert:
            # for bert, we take the representation of the first word
            ep_name = ep_toks + [type_tok]
            r.normalized_ep_names.append(Bertokens(ep_toks))
        else:
            ep_name = [type_tok] + ep_toks

        
        # if ep.edge is None:
        #     edge_name = ['<any-edge>']
        # else:
        #     edge_name = tokenize_func(
        #         ep.edge.name, ep.edge.unsplit_name)
        # ep_name += ['<edge-sep>'] + edge_name
        r.ep_names.append(ep_name)

        edge_id = None if ep.edge is None else ep.edge.id
        r.ep_to_edge[str(i)] = edge_id
        if edge_id is not None:
            eps = r.edge_to_eps.setdefault(str(edge_id), [])
            eps.append(i)
        if last_edge_id != edge_id:
            r.edge_bounds.append(i)
            last_edge_id = edge_id

        # if column.foreign_key_for is not None:
        #     r.foreign_keys[str(column.id)] = column.foreign_key_for.id
        #     r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.edge_bounds.append(len(schema.edge_properties))
    assert len(r.edge_bounds) == len(schema.edges) + 1

    for i, edge in enumerate(schema.edges):
        edge_toks = tokenize_func(
            edge.name, edge.unsplit_name)
        r.edge_names.append(edge_toks)
        if bert:
            r.normalized_edge_names.append(Bertokens(edge_toks))

    return r


class Neo4jEncoderV2Preproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000,
            include_table_name_in_column=True,
            word_emb=None,
            # count_tokens_in_word_emb_for_vocab=False,
            fix_issue_16_primary_keys=False,
            compute_sc_link=False,
            compute_cv_link=False):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = word_emb

        self.data_dir = os.path.join(save_path, 'enc')
        # self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.texts = collections.defaultdict(list)
        # self.db_path = db_path

        # self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        # self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        # self.vocab_word_freq_path = os.path.join(save_path, 'enc_word_freq.json')
        # self.vocab = None
        # self.counted_db_ids = set()
        self.preprocessed_schemas = {}

    def validate_item(self, item, schema, section):
        return True, None

    def add_item(self, item, schema, section, validation_info):
        preprocessed = self.preprocess_item(item, schema, validation_info)
        self.texts[section].append(preprocessed)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, schema, validation_info):
        question, question_for_copying = self._tokenize_for_copying(item['question_toks'], item['question'])
        preproc_schema = self._preprocess_schema(schema)
        if self.compute_sc_link:
            assert preproc_schema.np_names[0][0].startswith("<type:")
            np_names_without_types = [np[1:] for np in preproc_schema.np_names]
            sc_link_node = compute_schema_linking(question, np_names_without_types, preproc_schema.node_names, 'node')

            if preproc_schema.ep_names:
                # print(preproc_schema.ep_names)
                assert preproc_schema.ep_names[0][0].startswith("<type:")
                ep_names_without_types = [ep[1:] for ep in preproc_schema.ep_names]
                sc_link_edge = compute_schema_linking(question, ep_names_without_types, preproc_schema.edge_names, 'edge')
            else:
                sc_link_edge = {"q_ep_match": {}, "q_edge_match": {}}
        else:
            sc_link_node = {"q_np_match": {}, "q_node_match": {}}
            sc_link_edge = {"q_ep_match": {}, "q_edge_match": {}}

        if self.compute_cv_link:
            pass
            # cv_link = compute_cell_value_linking(question, schema)
        else:
            cv_link = {"num_date_match": {}, "cell_match": {}}
        return {
            'raw_question': item['question'],
            'db_id': schema.db_id,
            'question': question,
            'question_for_copying': question_for_copying,
            'sc_link_node': sc_link_node,
            'sc_link_edge': sc_link_edge,
            'cv_link': cv_link,

            'nps': preproc_schema.np_names,
            'nodes': preproc_schema.node_names,
            'node_bounds': preproc_schema.node_bounds,
            'np_to_node': preproc_schema.np_to_node,
            'node_to_nps': preproc_schema.node_to_nps,

            'eps': preproc_schema.ep_names,
            'edges': preproc_schema.edge_names,
            'edge_bounds': preproc_schema.edge_bounds,
            'ep_to_edge': preproc_schema.ep_to_edge,
            'edge_to_eps': preproc_schema.edge_to_eps,

            'node_point_to': preproc_schema.node_point_to,
            'node_be_point_to': preproc_schema.node_be_point_to,
        }

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # self.vocab = self.vocab_builder.finish()
        # print(f"{len(self.vocab)} words in vocab")
        # self.vocab.save(self.vocab_path)
        # self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '_schema-linking.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self, sections):
        # self.vocab = vocab.Vocab.load(self.vocab_path)
        # self.vocab_builder.load(self.vocab_word_freq_path)
        for section in sections:
            self.texts[section] = []
            with open(os.path.join(self.data_dir, section + '_schema-linking.jsonl'), 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        self.texts[section].append(json.loads(line))

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

