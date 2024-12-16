
# DAIL-Cypher Project

This project aims to support interaction with graph databases via natural language processing, enabling the generation of Cypher queries from natural language. It also integrates large language models (LLMs) for efficient task execution. Below is a detailed explanation of the project files and their functions.

---

## File Directory and Function Description

### 1. Data Processing Module

#### `\dataset\neo4j\normalized_data_structure.py`
- **Function**: Processes datasets containing `Question-Cypher` pairs.
- **Objective**: Converts the dataset into a normalized data structure.

#### `\data_proprocess.py`
- **Function**: Performs the `linking process` and `schema simplification`.
- **Objective**: Optimizes the data preprocessing workflow to provide necessary input data for `prompt generation`.

---

### 2. Prompts Generation Module

#### `\generate_questions.py`
- **Function**: Generates questions for LLMs and saves them as tasks.
- **Example Script**: `scripts/generate_questions.sh`
- **Input Parameters**:

  ```plaintext
  --data_type: Data type, supports "neo4j", "realistic", "bird".
  --split: Data split, supports "train", "test".
  --k_shot: Number of examples, default is 0.
  --prompt_repr: Prompt representation type, supports various modes (e.g., code, text).
  --example_type: Example type, either questions or complete examples.
  --selector_type: Example selection strategy, e.g., cosine similarity, random selection, Euclidean distance.
  --max_seq_len: Maximum sequence length, default is 2048.
  --max_ans_len: Maximum answer length, default is 200.
  --tokenizer: Tokenizer used, default is "gpt-3.5-turbo".
  --scope_factor: Scope multiplier, default is 100.
  ```

---

### 3. LLM Module

#### `\ask_llm.py`
- **Function**: Uses the output from `generate_questions.sh` as prompts to query the LLM.
- **Example Script**: `scripts/ask_llm.sh`
- **Input Parameters**:

  ```plaintext
  --question: Path to the question file.
  --openai_api_key: OpenAI API key.
  --openai_group_id: OpenAI group ID.
  --model: Model selection, supports various LLMs (e.g., LLAMA series).
  --start_index, --end_index: Query range.
  --temperature: Generation temperature, default is 0.
  --mini_index_path: Sub-index path.
  --batch_size: Batch size, default is 1.
  --n: Size of the self-consistency set, default is 5.
  --db_dir: Database directory.
  ```

---

### 4. Fine-Tuning Module

#### `\finetuning\finetuning.py`
- **Function**: Adjusts schema path and `questions` file path for model fine-tuning.

#### `\finetuning\lora_inference.py`
- **Function**: Uses the fine-tuned model for inference on the test set.

---

### 5. Evaluation Module

#### `\eval\data_postprocess_cypher.py`
- **Function**: Post-processes model output.

#### `\eval\eval.py`
- **Function**: Calculates execution similarity and exact match scores.

---

## Unmodified DAIL-SQL Source Code

The following files are inherited from DAIL-SQL and remain unmodified:
- `\prompt`
- `\utils\linking_utils\serialization.py`
- `\utils\linking_utils\abstract_preproc.py`
- `\utils\linking_utils\corenlp.py`
- `\utils\pretrained_embeddings.py`

---

## Runtime Environment and Dependencies

- Python Version: >= 3.8
- Refer to `requirements.txt`
