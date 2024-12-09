import argparse
import os 
import json

import openai
from tqdm import tqdm

from llm.ask_llm import init_chatgpt, ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader
from llm.llama3 import llama3
from llm.qwen import qwen
from llm.deepseek import deepseek_coder_v2

from utils.ebsc import get_cyphers
from utils.postprocess import extract_cypher_queries

QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default="org-ktBefi7n9aK7sZjwc2R9G1Wo")
    parser.add_argument("--model", type=str, choices=[LLM.TEXT_DAVINCI_003, 
                                                      LLM.GPT_35_TURBO,
                                                      LLM.GPT_35_TURBO_0613,
                                                      LLM.TONG_YI_QIAN_WEN,
                                                      LLM.GPT_35_TURBO_16K,
                                                      LLM.GPT_4,
                                                      LLM.META_LLAMA_3_8B,
                                                      LLM.LLAMA_3_8B,
                                                      LLM.LLAMA_2_13B,
                                                      LLM.QWEN_2_7B,
                                                      LLM.QWEN_PLUS,
                                                      LLM.DEEP_SEEK],
                        default=LLM.GPT_35_TURBO)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mini_index_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set")
    parser.add_argument("--db_dir", type=str, default="NA")
    args = parser.parse_args()

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"

    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    # init openai api
    if args.model == LLM.META_LLAMA_3_8B:
        llama3.init_model()
        model = llama3
    elif args.model == LLM.QWEN_2_7B:
        qwen.init_model()
        model = qwen
        # print(type(model))
    elif args.model == LLM.DEEP_SEEK:
        deepseek_coder_v2.init_model()
        model = deepseek_coder_v2
    else:
        init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)
        model = args.model


    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    if args.mini_index_path:
        mini_index = json.load(open(args.mini_index_path, 'r'))
        questions = [questions[i] for i in mini_index]
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    token_cnt = 0
    with open(out_file, mode) as f:
        for i, batch in enumerate(tqdm(question_loader)):
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            try:
                res = ask_llm(args.model, model, batch, args.temperature, args.n)
            except openai.error.InvalidRequestError:
                print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                res = ""

            # parse result
            token_cnt += res["total_tokens"]
            if args.n == 1:
                cypher = res["response"]
                print(res["response"])
                # remove \n and extra spaces
                cypher = " ".join(cypher.replace("\n", " ").split())
                cypher = extract_cypher_queries(cypher)
                print(cypher)
                # python version should >= 3.8
                if cypher.startswith("MATCH"):
                    f.write(cypher + "\n")
                elif cypher.startswith(" ("):
                    f.write("MATCH" + cypher + "\n")
                elif cypher.startswith("("):
                    f.write("MATCH " + cypher + "\n")
                else:
                    f.write("MATCH (" + cypher + "\n")
            else:
                results = []
                cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                for cyphers, db_id in zip(res["response"], cur_db_ids):
                    processed_cyphers = []
                    for cypher in cyphers:
                        cypher = " ".join(cypher.replace("\n", " ").split())
                        cypher = extract_cypher_queries(cypher)
                        if cypher.startswith("MATCH"):
                            pass
                        elif cypher.startswith(" ("):
                            cypher = "MATCH" + cypher
                        elif cypher.startswith("("):
                            cypher = "MATCH " + cypher
                        else:
                            cypher = "MATCH (" + cypher
                        processed_cyphers.append(cypher)
                    result = {
                        'db_id': db_id,
                        'p_cyphers': processed_cyphers
                    }
                    final_cyphers = get_cyphers([result], args.n, args.db_dir)

                    for cypher in final_cyphers:
                        f.write(cypher + "\n")

