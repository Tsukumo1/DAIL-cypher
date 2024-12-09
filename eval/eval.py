import re
from neo4j import GraphDatabase
from tqdm import tqdm

class CypherEvaluator:
    def __init__(self):
        self.clause_keywords = ['MATCH', 'WHERE', 'WITH', 'RETURN', 'ORDER BY', 'LIMIT', 'CREATE', 'MERGE', 'SET', 'DELETE']
    
    def parse_cypher(self, query):
        query = query.lower()
        parsed = {keyword.lower(): None for keyword in self.clause_keywords}
        parts = re.split(r'\b(' + '|'.join(map(re.escape, self.clause_keywords)).lower() + r')\b', query)
        current_clause = None
        for part in parts:
            part = part.strip()
            if part in parsed:
                current_clause = part
            elif current_clause and part:
                parsed[current_clause] = part
        return parsed
    
    def normalize_identifiers(self, clause):
        return re.sub(r'(?<=\W)(\w+)(?=\W)', 'VAR', clause) if clause else None
    
    def eval_exact_match(self, pred, label):
        pred_parsed = self.parse_cypher(pred)
        label_parsed = self.parse_cypher(label)
        for clause in self.clause_keywords:
            clause = clause.lower()
            pred_clause = self.normalize_identifiers(pred_parsed[clause])
            label_clause = self.normalize_identifiers(label_parsed[clause])
            if pred_clause != label_clause:
                return 0
        return 1

class Neo4jDatabase:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def execute_query(self, query):
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                return [record for record in result]
            except Exception as e:
                return None

def clean_result(result):
    return sorted([str(item) for item in result])

def compare_results(result1, result2):
    if result1 is not None:
        return clean_result(result1) == clean_result(result2)
    else:
        return False

def read_file(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines()]

def parse_gold_line(line):
    parts = line.split('\t')
    return parts[0], parts[1]

def evaluate_cypher_queries(gold_queries, pred_queries):
    evaluator = CypherEvaluator()
    total_queries = len(gold_queries)
    exact_matches = 0
    for gold, pred in zip(gold_queries, pred_queries):
        exact_matches += evaluator.eval_exact_match(pred, gold)
    exact_score = exact_matches / total_queries
    return exact_score

def evaluate_execution(predict_file, gold_file, neo4j_uri):
    predictions = read_file(predict_file)
    golds = read_file(gold_file)

    total = len(predictions)
    correct = 0
    valid_queries = 0

    print(len(predictions))
    print(len(golds))

    with tqdm(total=total, desc="Evaluating execution") as pbar:
        for i in range(total):
            pred_cypher = predictions[i]
            gold_cypher, db_name = parse_gold_line(golds[i])

            db = Neo4jDatabase(uri=neo4j_uri, user=db_name, password=db_name, database=db_name)

            try:
                gold_result = db.execute_query(gold_cypher)
                if gold_result is None:
                    tqdm.write(f"Error executing gold query on line {i+1}. Skipping this query.")
                    pbar.update(1)
                    continue

                pred_result = db.execute_query(pred_cypher)
                # print(valid_queries)
                
                if compare_results(pred_result, gold_result):
                    correct += 1
                valid_queries += 1
            except Exception as e:
                tqdm.write(f"Error executing queries on line {i+1}: {e}")
            finally:
                db.close()
            
            pbar.update(1)

    accuracy = (correct / valid_queries * 100) if valid_queries > 0 else 0
    return accuracy, correct, valid_queries

def main():
    predict_file = 'path_to_predict_result'
    gold_file = 'path_to_gold'
    neo4j_uri = 'neo4j+s://demo.neo4jlabs.com'

    print("Starting evaluation...")


    # Evaluate execution similarity
    print("Evaluating execution similarity...")
    execution_accuracy, correct_executions, valid_executions = evaluate_execution(predict_file, gold_file, neo4j_uri)
    print(f"Execution similarity: {execution_accuracy:.2f}% ({correct_executions}/{valid_executions})")

    # Evaluate exact match
    print("\nEvaluating exact match...")
    gold_queries = [parse_gold_line(line)[0] for line in read_file(gold_file)]
    pred_queries = read_file(predict_file)
    
    total_queries = len(gold_queries)
    exact_matches = 0
    evaluator = CypherEvaluator()
    
    print(len(gold_queries))
    print(len(pred_queries))
    for gold, pred in tqdm(zip(gold_queries, pred_queries), total=total_queries, desc="Evaluating exact matches"):
        exact_matches += evaluator.eval_exact_match(pred, gold)
    
    exact_score = exact_matches / total_queries
    print(f"Exact match score: {exact_score:.2f} ({exact_matches}/{total_queries})")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
