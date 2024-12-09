import re

def extract_cypher_queries(cypher):
    
    pattern1 = r'```\s*(MATCH[\s\S]*?)```'

    pattern4 = r'```cypher\s*(MATCH[\s\S]*?)```'

    pattern2 = r':\s*(MATCH[\s\S]*?)(?=\n\S|$)'

    pattern3 = r':\s*(MATCH[\s\S]*?);'

    pattern5 = r':\s*(MATCH[\s\S]*?)This'

    pattern6 = r'\*/\s*(MATCH[\s\S]*?)(?=\n\S|$)'

    pattern7 = r'1\. (.*?)2\.'

    pattern8 = r'1\. (.*)$'

    cypher_queries = []

    i = cypher
    matches1 = re.findall(pattern1, i, re.IGNORECASE)
    matches2 = re.findall(pattern2, i, re.IGNORECASE)
    matches3 = re.findall(pattern3, i, re.IGNORECASE)
    matches4 = re.findall(pattern4, i, re.IGNORECASE)
    matches5 = re.findall(pattern5, i, re.IGNORECASE)
    matches6 = re.findall(pattern6, i, re.IGNORECASE)
    matches7 = re.findall(pattern7, i, re.IGNORECASE)
    matches8 = re.findall(pattern8, i, re.IGNORECASE)
        
    if matches1:
        str = matches1[0].rstrip('; ') 
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches4:
        str = matches4[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches3:
        str = matches3[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches5:
        str = matches5[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches2:
        str = matches2[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches6:
        str = matches6[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str+'\n')
    elif matches7:
        str = matches7[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append('MATCH ('+str+'\n')
    elif matches8:
        str = matches8[0].rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append('MATCH ('+str+'\n')
    else:
        str = i.rstrip('; ')
        str = str.replace(";", "")
        str = str.replace("`", "")
        cypher_queries.append(str)

    cypher_queries = [query.rstrip(';') for query in cypher_queries]

    cypher_queries = [query.rstrip('`') for query in cypher_queries]

    return cypher_queries[0]