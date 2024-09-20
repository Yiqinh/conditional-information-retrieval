# Function to get immediate children of a node
def get_immediate_children(node_id, linkage_matrix, n_samples):
    if node_id < n_samples:
        return []  # Leaf nodes have no children
    else:
        left_child = int(linkage_matrix[node_id - n_samples, 0])
        right_child = int(linkage_matrix[node_id - n_samples, 1])
        return [left_child, right_child]


# Function to get sample indices for each node
def get_cluster_indices(node_id, linkage_matrix, n_samples, node_samples):
    if node_id in node_samples:
        return node_samples[node_id]
    if node_id < n_samples:
        node_samples[node_id] = [node_id]
    else:
        left_child = int(linkage_matrix[node_id - n_samples, 0])
        right_child = int(linkage_matrix[node_id - n_samples, 1])
        left_indices = get_cluster_indices(left_child)
        right_indices = get_cluster_indices(right_child)
        node_samples[node_id] = left_indices + right_indices
    return node_samples[node_id]


def print_tree(node_id, level=0, max_nodes=None, nodes_printed=[0], node_summaries=None, n_samples=None):
    if node_summaries is None:
        node_summaries = {}

    if max_nodes is not None and nodes_printed[0] >= max_nodes:
        return
    indent = '    ' * level
    summary = node_summaries.get(node_id, "No summary available")
    print(f"{indent}- Node {node_id}: {summary}")
    nodes_printed[0] += 1
    if node_id < n_samples:
        # Leaf node; no further traversal
        return
    # Get immediate children
    children = get_immediate_children(node_id)
    for child_id in children:
        if max_nodes is not None and nodes_printed[0] >= max_nodes:
            return
        print_tree(child_id, level + 1, max_nodes, nodes_printed)


# Function to collect all nodes at a specific depth
def collect_nodes_at_depth_k(node_id, level=0, target_level=0, n_samples=None):
    """
    Recursively collects all nodes at a specific depth level.

    Parameters:
    - node_id (int): The ID of the current node.
    - level (int): The current depth level in the tree.
    - target_level (int): The depth level at which to collect nodes.

    Returns:
    - List of node IDs at the target depth level.
    """
    if level == target_level:
        return [node_id]
    if node_id < n_samples:
        return []  # Leaf node; no further traversal
    nodes = []
    children = get_immediate_children(node_id)
    for child_id in children:
        nodes.extend(collect_nodes_at_depth_k(child_id, level + 1, target_level))
    return nodes        


# Function to build the parent mapping
def build_parent_dict(n_samples):
    """
    Builds a dictionary that maps each node to its parent node.
    
    Returns:
    - parent_dict (dict): A mapping from child node IDs to parent node IDs.
    """
    parent_dict = {}
    # Iterate over each non-leaf node and its children
    for node_id in range(n_samples, 2 * n_samples - 1):
        children = get_immediate_children(node_id)
        for child_id in children:
            parent_dict[child_id] = node_id
    return parent_dict


# Function to trace a leaf node up to the root
def trace_leaf_to_root(leaf_node_id, parent_dict):
    """
    Traces the path from a leaf node up to the root node.
    
    Parameters:
    - leaf_node_id (int): The ID of the leaf node (must be less than n_samples).
    - parent_dict (dict): A mapping from child node IDs to parent node IDs.
    
    Returns:
    - path (list): A list of node IDs from the leaf to the root.
    """
    path = [leaf_node_id]
    current_node_id = leaf_node_id
    while current_node_id in parent_dict:
        parent_node_id = parent_dict[current_node_id]
        path.append(parent_node_id)
        current_node_id = parent_node_id
    return path


import matplotlib.pyplot as plt
import networkx as nx


def hierarchy_pos(graph, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Position nodes in a hierarchical layout.
    
    :param graph: NetworkX graph object.
    :param root: The root node of the graph.
    :param width: Total width of the plot.
    :param vert_gap: Vertical gap between levels.
    :param vert_loc: Initial vertical location of the root.
    :param xcenter: Horizontal center of the root node.
    :return: A dictionary of positions keyed by node.
    """
    pos = _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(graph, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    """
    Helper function for computing hierarchical layout.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
        
    children = list(graph.neighbors(root))
    
    if not isinstance(graph, nx.DiGraph) and parent is not None:
        children.remove(parent)  # Avoid cycles in undirected graph
    
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(graph, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, 
                                 xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    
    return pos


import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

def plot_hierarchical_tree(tree, depth=None, figsize=(8, 6)):
    """
    Plot a hierarchical tree using NetworkX and Matplotlib.
    
    :param tree: Dictionary representing the tree structure.
    :param depth: Maximum depth to traverse. If None, traverse entire tree.
    :param figsize: Tuple of figure width and height in inches.
    """
    def add_edges_iteratively(graph, root, depth=None):
        """
        Iteratively add edges to the graph from a tree structure.
        
        :param graph: NetworkX graph object.
        :param root: Root node of the tree.
        :param depth: Maximum depth to traverse. If None, traverse entire tree.
        """
        stack = [(root, None, 0)]  # (current_node, parent_node, current_depth)
        while stack:
            node, parent, current_depth = stack.pop()
            if parent is not None:
                graph.add_edge(parent, node["node"])
            
            if depth is not None and current_depth >= depth:
                continue
            
            for child in node.get("children", []):
                stack.append((child, node["node"], current_depth + 1))
    
    graph = nx.DiGraph()
    
    # Iteratively add nodes and edges to the graph
    add_edges_iteratively(graph, tree, depth=depth)
    
    # Get hierarchical positions for each node
    pos = hierarchy_pos(graph, tree["node"])
    
    # Plot the tree structure
    plt.figure(figsize=figsize)
    nx.draw(graph, pos, with_labels=False, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", arrows=False)
    
    # Adjust text positions to avoid overlap
    texts = []
    for node, (x, y) in pos.items():
        text = plt.text(x, y, node, fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        texts.append(text)
    
    adjust_text(texts, only_move={'points': 'y', 'text': 'xy'})
    
    # Show plot
    plt.show()

# Example tree structure
tree = {
    "node": "a",
    "children": [
        {"node": "b", "children": [
            {"node": "d", "children": []},
            {"node": "e", "children": []}
        ]},
        {"node": "c", "children": [
            {"node": "f", "children": []}
        ]}
    ]
}


def print_tree(node, prefix="", is_last=True):
    """
    Prints a tree represented as a dictionary in ASCII format.
    
    :param node: Dictionary with keys "node" (the current node's name) and "children" (list of child nodes).
    :param prefix: String to define the current prefix for the branch.
    :param is_last: Boolean indicating if the current node is the last child of its parent.
    """
    connector = "└── " if is_last else "├── "
    print(prefix + connector + node["node"])

    # Update the prefix for child nodes
    prefix += "    " if is_last else "│   "

    # Recursively print each child node
    children = node.get("children", [])
    for i, child in enumerate(children):
        is_last_child = i == (len(children) - 1)
        print_tree(child, prefix, is_last_child)


import plotly.graph_objs as go
from igraph import Graph, EdgeSeq


# Function to convert input tree into a list of edges
def parse_tree_to_edges(tree, parent=None):
    """
    Convert input tree into a list of edges.
    
    :param tree: Dictionary representing the tree structure.
    :param parent: Parent node of the current node.
    :return: List of edges.
    """
    if tree is None:
        return None
    if "node" not in tree:
        return None
    edges = []
    node = tree["node"]
    if parent:
        edges.append((parent, node))
    for child in (tree.get("children", []) or []):  # Use .get() with a default empty list
        if child is not None:
            output = parse_tree_to_edges(child, node)
            if output is not None:
                edges.extend(output)
    return edges


def plot_tree_with_plotly(tree, figsize=(800, 600)):
    """
    Plot a tree using Plotly.
    
    :param tree: Dictionary representing the tree structure.
    :param figsize: Tuple of figure width and height in pixels.
    :return: None
    """
    def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
        """
        Create annotations for the plot.
        
        :param pos: Positions of the nodes.
        :param text: Labels for the nodes.
        :param font_size: Font size for the annotations.
        :param font_color: Font color for the annotations.
        :return: List of annotations.
        """
        if len(pos) != len(text):
            raise ValueError("Position and text lists must have the same length.")
        
        annotations = []
        for k in range(len(pos)):
            annotations.append(
                dict(
                    text=text[k],
                    x=pos[k][0], y=2 * M - pos[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False
                )
            )
        return annotations

    # Parse the tree and extract edges
    edges = parse_tree_to_edges(tree)
    nodes = set([edge[0] for edge in edges] + [edge[1] for edge in edges])
    node_indices = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices for igraph

    # Create a graph using igraph
    G = Graph()
    G.add_vertices(len(nodes))
    G.add_edges([(node_indices[edge[0]], node_indices[edge[1]]) for edge in edges])

    # Layout using the Reingold-Tilford algorithm
    layout = G.layout('rt')
    positions = {k: layout[k] for k in range(len(nodes))}
    Y = [layout[k][1] for k in range(len(nodes))]
    M = max(Y)

    # Prepare edges for Plotly
    edge_seq = EdgeSeq(G)
    plot_edges = [e.tuple for e in edge_seq]

    # X and Y coordinates for nodes
    Xn = [positions[k][0] for k in range(len(nodes))]
    Yn = [2 * M - positions[k][1] for k in range(len(nodes))]

    # X and Y coordinates for edges
    Xe = []
    Ye = []
    for edge in plot_edges:
        Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
        Ye += [2 * M - positions[edge[0]][1], 2 * M - positions[edge[1]][1], None]

    # Labels for nodes
    labels = list(node_indices.keys())

    # Create Plotly traces for edges and nodes
    lines = go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgb(210,210,210)', width=1),
        hoverinfo='none'
    )

    dots = go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        marker=dict(
            symbol='circle',  # Changed from 'dot' to 'circle'
            size=18,
            color='#6175c1',
            line=dict(color='rgb(50,50,50)', width=1)
        ),
        text=labels,
        hoverinfo='text',
        opacity=0.8
    )

    # Axis configuration
    axis = dict(
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    # Layout for Plotly plot
    width, height = figsize  # Now in pixels
    layout = dict(
        title='Tree with Reingold-Tilford Layout',
        annotations=make_annotations(positions, labels),
        font=dict(size=12),
        showlegend=False,
        xaxis=dict(axis),
        yaxis=dict(axis),
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode='closest',
        plot_bgcolor='rgb(248,248,248)',
        width=width,
        height=height
    )

    # Plot the tree
    data = [lines, dots]
    fig = dict(data=data, layout=layout)
    fig['layout'].update(annotations=make_annotations(positions, labels))
    return fig
    # in notebook, use:
    # 
    # Use plotly offline or online depending on the setup
    # import plotly.offline as pyo
    # pyo.plot(fig)