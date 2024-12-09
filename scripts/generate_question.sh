
cd ../third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd ../../


python generate_question.py \
--data_type neo4j \
--split test \
--tokenizer gpt-3.5-turbo \
--max_seq_len 4096 \
--prompt_repr Cypher \
--k_shot 9 \
--example_type QA \
--selector_type  EUCDISQUESTIONMASK