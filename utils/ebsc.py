import random
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Set
import tqdm



def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False



from neo4j import GraphDatabase
def get_exec_output(
        uri, user, password, database,
        cypher
): 
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        try:
            result = session.run(cypher)
            return "0", [record for record in result]
        except Exception as e:
            return "exception", []


def get_cyphers(results, select_number):
    db_ids = []
    all_p_cyphers = []
    for item in results:
        p_cyphers = []
        db_ids.append(item['db_id'])
        for i, x in enumerate(item['p_cyphers']):
            p_cyphers.append(x)
            if i+1 == select_number:
                break
        all_p_cyphers.append(p_cyphers)
    chosen_p_cyphers = []
    for i, db_id in enumerate(tqdm.tqdm(db_ids)):
        p_cyphers = all_p_cyphers[i]
        cluster_cypher_list = []
        map_cypher2denotation = {}
        for cypher in p_cyphers:
            flag, denotation = get_exec_output(
                'neo4j+s://demo.neo4jlabs.com',
                db_id,db_id,db_id,
                cypher
            )
            if flag == "exception":
                continue
            map_cypher2denotation[cypher] = denotation
            denotation_match = False

            for id, cluster in enumerate(cluster_cypher_list):
                center_cypher = cluster[0]
                if result_eq(map_cypher2denotation[center_cypher], denotation, False):
                    cluster_cypher_list[id].append(cypher)
                    denotation_match = True
                    break
            if not denotation_match:
                cluster_cypher_list.append([cypher])
        cluster_cypher_list.sort(key=lambda x: len(x), reverse=True)
        if not cluster_cypher_list:
            chosen_p_cyphers.append(p_cyphers[0])
        else:
            largest_cluster = cluster_cypher_list[0]
            cypher_counts = {}
            for cypher in largest_cluster:
                cypher_counts[cypher] = cypher_counts.get(cypher, 0) + 1
            most_common_cypher = max(cypher_counts, key=cypher_counts.get)
            chosen_p_cyphers.append(most_common_cypher)

    return chosen_p_cyphers