cd ../third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd ../../

python ask_llm.py \
--openai_api_key  your_key \
--model deepseek_coder_v2 \
--n 1 \
--temperature 0.01 \
--question path_to_question
