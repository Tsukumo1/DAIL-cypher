import argparse
import json
import os
from tqdm import tqdm

from utils.linking_process import Neo4jEncoderV2Preproc
from utils.pretrained_embeddings import GloVe
from utils.datasets.cypher import load_schemas
# from dataset.process.preprocess_kaggle import gather_questions


def schema_linking_producer(test, train, table, dataset_dir, compute_cv_link=False):

    # load data
    test_data = json.load(open(os.path.join(dataset_dir, test)))
    train_data = json.load(open(os.path.join(dataset_dir, train)))

    # load schemas
    schemas= load_schemas([os.path.join(dataset_dir, table)])
    print('already load schemas')

    word_emb = GloVe(kind='42B', lemmatize=True)
    linking_processor = Neo4jEncoderV2Preproc(dataset_dir,
            min_freq=4,
            max_count=5000,
            include_table_name_in_column=False,
            word_emb=word_emb,
            fix_issue_16_primary_keys=True,
            compute_sc_link=True,
            compute_cv_link=compute_cv_link)

    # build schema-linking
    for data, section in zip([test_data, train_data],['test', 'train']):
        for item in tqdm(data, desc=f"{section} section linking"):
            db_id = item["db_id"]
            schema = schemas[db_id]
            to_add, validation_info = linking_processor.validate_item(item, schema, section)
            if to_add:
                linking_processor.add_item(item, schema, section, validation_info)

    # save
    linking_processor.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset/neo4j")
    parser.add_argument("--data_type", type=str, choices=["neo4j", "spider"], default="neo4j")
    args = parser.parse_args()

    data_type = args.data_type
    if data_type == "neo4j":
        neo4j_dir = args.data_dir
        # schema-linking between questions and databases for neo4j
        neo4j_dev = "test.json"
        neo4j_train = 'train.json'
        neo4j_table = 'schemas.json'
        schema_linking_producer(neo4j_dev, neo4j_train, neo4j_table, neo4j_dir)
   