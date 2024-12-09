import json
import re
from copy import copy
from pathlib import Path
from typing import List, Dict

import attr
import torch
import networkx as nx
from tqdm import tqdm


@attr.s
class NodeProperties:
    id = attr.ib()
    node = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()


@attr.s
class Node:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    node_properties = attr.ib(factory=list)
    point_to = attr.ib(factory=list) ## [[target, edge_id], ]
    be_point_to = attr.ib(factory=list) ## [[root, edge_id], ]


@attr.s
class Schema:
    db_id = attr.ib()
    nodes = attr.ib()
    node_properties = attr.ib()
    edges = attr.ib()
    edge_properties = attr.ib()
    relation_graph = attr.ib()
    orig = attr.ib()
    connection = attr.ib(default=None)

@attr.s
class Edge:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    edge_properties = attr.ib(factory=list)

@attr.s
class EdgeProperties:
    id = attr.ib()
    edge = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()

def load_schemas(paths):
    schemas = {}

    for path in paths:
        schema_dicts = json.load(open(path))
        for schema_dict in schema_dicts:
            nodes = tuple(
                Node(
                    id=i,
                    name=name.split(),
                    unsplit_name=name,
                    orig_name=orig_name,
                )
                for i, (name, orig_name) in enumerate(zip(
                    schema_dict['Node'], schema_dict['Node_original']))
            )
            node_properties = tuple(
                NodeProperties(
                    id=i,
                    node=nodes[node_id] if node_id >= 0 else None,
                    name=node_property.split(),
                    unsplit_name=node_property,
                    orig_name=node_property_original,
                    type=node_property_type,
                )
                for i, (node_property, (node_id, node_property_original), node_property_type) in enumerate(zip(
                    schema_dict['Node_properties'],
                    schema_dict['Node_properties_original'],
                    schema_dict['Node_properties_type']))
            )

            # Link node_properties to nodes
            for node_property in node_properties:
                if node_property.node:
                    node_property.node.node_properties.append(node_property)


            # Check if Edge data is available
            if schema_dict['Edge'] and schema_dict['Edge_original']:
                edges = tuple(
                    Edge(
                        id=i,
                        name=name.split(),
                        unsplit_name=name,
                        orig_name=orig_name,
                    )
                    for i, (name, orig_name) in enumerate(zip(
                        schema_dict['Edge'], schema_dict['Edge_original']))
                )
            else:
                edges = tuple()  # Empty tuple if no Edge data

            # Check if Edge properties data is available
            if (schema_dict['Edge_properties'] and 
                schema_dict['Edge_properties_original'] and 
                schema_dict['Edge_properties_type']):
                edge_properties = tuple(
                    EdgeProperties(
                        id=i,
                        edge=edges[edge_id] if edges and edge_id >= 0 and edge_id < len(edges) else None,
                        name=edge_property.split(),
                        unsplit_name=edge_property,
                        orig_name=edge_property_original,
                        type=edge_property_type,
                    )
                    for i, (edge_property, (edge_id, edge_property_original), edge_property_type) in enumerate(zip(
                        schema_dict['Edge_properties'],
                        schema_dict['Edge_properties_original'],
                        schema_dict['Edge_properties_type']))
                )
            else:
                edge_properties = tuple()  # Empty tuple if no Edge properties data

            # Link edge_properties to edges only if both exist
            if edges and edge_properties:
                for edge_property in edge_properties:
                    if edge_property.edge:
                        edge_property.edge.edge_properties.append(edge_property)

            # relation
            relation_graph = nx.DiGraph()

            for (origin, relation, target, origin_id, target_id) in schema_dict['relationship']:
                # Register target
                source_node = nodes[origin_id]
                target_node = nodes[target_id]
                source_node.point_to.append([target_node.id, relation])
                target_node.be_point_to.append([source_node.id, relation])

                relation_graph.add_edge(
                    source_node.id,
                    target_node.id,
                    nodes_id = (origin_id, target_id),
                    nodes = (source_node, target_node),
                    relation = relation
                    )
                

            db_id = schema_dict['db_id']
            assert db_id not in schemas
            schemas[db_id] = Schema(db_id, nodes, node_properties, edges, edge_properties, relation_graph, schema_dict)
            

    return schemas

