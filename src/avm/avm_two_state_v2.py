import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
"""
Adaptive Voter Model (AVM) Implementation

This module implements the Adaptive Voter Model with homophily-driven rewiring
for studying opinion dynamics and network evolution.

Author: [Your Name]
License: [Your License]
"""

import numpy as np
import networkx as nx
from copy import deepcopy  
from scipy.integrate import solve_ivp
from numba import njit

# ============================================================================
# NETWORK ANALYSIS FUNCTIONS
# ============================================================================

@njit
def calculate_average_clustering_coefficient(adj_matrix):
    """
    Calculate the average clustering coefficient of the graph.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        
    Returns:
        float: Average clustering coefficient across all nodes.
    """
    num_nodes = adj_matrix.shape[0]
    total_coeff = 0.0
    count = 0
    
    for i in range(num_nodes):
        # Get neighbors of node i
        neighbors = []
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                neighbors.append(j)
        
        k = len(neighbors)
        if k < 2:
            continue  # clustering coefficient is 0 for nodes with degree < 2
            
        # Count number of edges between neighbors
        edges_between_neighbors = 0
        for a in range(k):
            for b in range(a+1, k):
                if adj_matrix[neighbors[a], neighbors[b]] == 1:
                    edges_between_neighbors += 1
        
        possible_edges = k * (k - 1) / 2
        coeff = edges_between_neighbors / possible_edges
        total_coeff += coeff
        count += 1
    
    if count == 0:
        return 0.0
    return total_coeff / count

@njit
def calculate_global_clustering_coefficient(adj_matrix):
    """
    Calculate the global clustering coefficient using matrix operations.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        
    Returns:
        float: Global clustering coefficient (ratio of triangles to triplets).
    """
    num_nodes = adj_matrix.shape[0]
    triangles = 0
    triplet_sum = 0
    
    for i in range(num_nodes):
        # Compute degree k_i (number of neighbors)
        k_i = 0
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                k_i += 1
        triplet_sum += k_i * (k_i - 1)
        
        # Count triangles involving node i, j, k (i < j < k to avoid duplicates)
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                for k in range(num_nodes):
                    if adj_matrix[j, k] == 1 and adj_matrix[k, i] == 1:
                        triangles += 1
    
    # Each triangle is counted 6 times (permutations of i,j,k), so divide by 6
    triangles = triangles // 6
    
    # Each triplet is counted twice (once for each direction), so divide by 2
    triplet_sum = triplet_sum // 2
    
    if triplet_sum == 0:
        return 0.0
    
    return triangles / triplet_sum

@njit
def get_largest_component_size(adj_matrix):
    """
    Calculate the size of the largest connected component in an undirected graph.
    
    Uses depth-first search (DFS) to identify all connected components and returns
    the size of the largest one.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        
    Returns:
        int: Size of the largest connected component.
    
    Returns:
        largest_cluster (int): Size of the largest connected component.
    """
    num_nodes = adj_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=np.int8)  # Smaller dtype
    #visited = np.zeros(num_nodes, dtype=np.int32)
    
    def dfs(node):
        """Depth-First Search to find the size of the connected component."""
        stack = [node]
        component_size = 0
        
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = 1
                component_size += 1
                # Explore all neighbors
                neighbors = np.where(adj_matrix[current] == 1)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
        
        return component_size
    
    largest_cluster = 0
    for node in range(num_nodes):
        if not visited[node]:
            cluster_size = dfs(node)
            if cluster_size > largest_cluster:
                largest_cluster = cluster_size
    
    return largest_cluster

# ============================================================================
# NETWORK REWIRING FUNCTIONS
# ============================================================================

@njit
def rewire(nodes, adj_matrix, opinions, directed=False, search='global', force_replacement=True, homo_prob=1.0):
    """
    Main rewiring function that dispatches to appropriate implementation.
    
    This function handles both directed and undirected rewiring by calling
    the appropriate specialized function based on the network type.
    
    Parameters:
        nodes (np.ndarray): Array of node indices to choose from.
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        opinions (np.ndarray): Array of opinion values for each node.
        directed (bool): Whether to use directed or undirected rewiring.
        search (str): Search strategy - 'global' or 'local'.
        force_replacement (bool): Whether to force rewiring regardless of satisfaction.
        homo_prob (float): Homophily probability for rewiring (undirected only).
    
    Returns:
        tuple: (edges_to_remove, edges_to_add) as arrays of edge coordinates.
    """
    i = nodes[np.random.randint(len(nodes))]
    if directed == False:
        edge_to_remove, edge_to_add = rewire_single_u(i, adj_matrix, opinions, search, force_replacement, homo_prob)
    else:
        edge_to_remove, edge_to_add = rewire_single_d(i, adj_matrix, opinions, search, force_replacement)
    return edge_to_remove, edge_to_add

@njit
def rewire_single_d(i, adj_matrix, opinions, search='global', force_replacement=True):
    """
    Directed rewiring for node i, focusing only on in-degrees.

    Parameters:
        i (int): Node to rewire.
        adj_matrix (np.ndarray): Adjacency matrix.
        opinions (np.ndarray): Array of opinions for each node.
        search (str): 'global' or 'local' search strategy.
        force_replacement (bool): Force rewiring even if opinions match.

    Returns:
        tuple: (edges_to_remove, edges_to_add)
    """
    num_nodes = adj_matrix.shape[0]
    max_edges = num_nodes - 1
    edges_to_remove = np.zeros((1, 2), dtype=np.int32)
    edges_to_add = np.zeros((1, 2), dtype=np.int32)
    edges_to_remove.fill(-1)
    edges_to_add.fill(-1)

    # 1. Identify predecessors (incoming edges)
    predecessors = np.where(adj_matrix[:, i] == 1)[0]

    # 2. Check if node is isolated or fully connected
    if len(predecessors) == 0 or len(predecessors) == max_edges:
        return edges_to_remove, edges_to_add

    # 3. Check if all predecessors have the same opinion
    target_opinion = opinions[i]
    if not force_replacement:
        all_same_opinion = True
        for pred in predecessors:
            if opinions[pred] != target_opinion:
                all_same_opinion = False
                break
        if all_same_opinion:
            return edges_to_remove, edges_to_add

    # 4. Get non-predecessors for candidate selection
    all_nodes = np.arange(num_nodes)
    non_predecessors = np.setdiff1d(all_nodes, np.append(predecessors, i))

    # 5. Candidate Selection
    candidates = []
    if search == 'global':
        # Global search for nodes with the same opinion
        for node in non_predecessors:
            if opinions[node] == target_opinion:
                candidates.append(node)

    elif search == 'local':
        # Local search among predecessors' neighbors
        neighbors_of_predecessors = []
        for pred in predecessors:
            second_neighbors = np.where(adj_matrix[pred] == 1)[0]
            for second_neighbor in second_neighbors:
                if second_neighbor != i and second_neighbor not in predecessors:
                    neighbors_of_predecessors.append(second_neighbor)
        
        # Filter by opinion
        for node in neighbors_of_predecessors:
            if opinions[node] == target_opinion:
                candidates.append(node)
    else:
        return edges_to_remove, edges_to_add  # Invalid search, do nothing

    if len(candidates) == 0:
        return edges_to_remove, edges_to_add

    # 6. Select Edge to Remove (Incoming Edge)
    if not force_replacement:
        mismatched_predecessors = []
        for pred in predecessors:
            if opinions[pred] != target_opinion:
                mismatched_predecessors.append(pred)
        
        if len(mismatched_predecessors) == 0:
            return edges_to_remove, edges_to_add
        
        to_remove = mismatched_predecessors[np.random.randint(len(mismatched_predecessors))]
    else:
        to_remove = predecessors[np.random.randint(len(predecessors))]
    
    edges_to_remove[0] = (to_remove, i)

    # 7. Select Edge to Add (New Incoming Edge)
    new_node = candidates[np.random.randint(len(candidates))]
    
    # Ensure directed edge does not already exist and no self-loop
    if adj_matrix[new_node, i] == 0 and new_node != i:
        edges_to_add[0] = (new_node, i)

    return edges_to_remove, edges_to_add

@njit
def rewire_single_u(i, adj_matrix, opinions, search='global', force_replacement=True, homo_prob=0.0):
    """
    Undirected rewiring for node i with probabilistic homophily (Numba-compatible).

    Parameters:
        i (int): Node to rewire.
        adj_matrix (np.ndarray): Adjacency matrix.
        opinions (np.ndarray): Array of opinions for each node.
        search (str): 'global' or 'local' search strategy.
        force_replacement (bool): Force rewiring even if opinions match.
        homo_prob (float): Probability of rewiring to a node with the same opinion.

    Returns:
        tuple: (edges_to_remove, edges_to_add)
    """
    n = adj_matrix.shape[0]
    max_edges = n - 1

    edges_to_remove = np.full((2, 2), -1, dtype=np.int32)
    edges_to_add = np.full((2, 2), -1, dtype=np.int32)

    # Identify neighbors
    current_neighbors = np.where(adj_matrix[i] == 1)[0]
    if len(current_neighbors) == 0 or len(current_neighbors) == max_edges:
        return edges_to_remove, edges_to_add

    target_opinion = opinions[i]

    # Skip rewiring if not forcing replacement and all neighbors have same opinion
    if not force_replacement:
        same = True
        for nb in current_neighbors:
            if opinions[nb] != target_opinion:
                same = False
                break
        if same:
            return edges_to_remove, edges_to_add

    # Identify non-neighbors
    all_nodes = np.arange(n)
    mask = np.ones(n, dtype=np.bool_)
    mask[current_neighbors] = False
    mask[i] = False
    non_neighbors = all_nodes[mask]

    # Candidate pools
    candidates_same = np.empty(0, dtype=np.int32)
    candidates_diff = np.empty(0, dtype=np.int32)

    if search == 'global':
        same_tmp = []
        diff_tmp = []
        for node in non_neighbors:
            if opinions[node] == target_opinion:
                same_tmp.append(node)
            else:
                diff_tmp.append(node)
        if len(same_tmp) > 0:
            candidates_same = np.array(same_tmp, dtype=np.int32)
        if len(diff_tmp) > 0:
            candidates_diff = np.array(diff_tmp, dtype=np.int32)



    elif search == 'local':
        same_tmp = []
        diff_tmp = []
        for nb in current_neighbors:
            neigh_of_nb = np.where(adj_matrix[nb] == 1)[0]
            for node in neigh_of_nb:
                if node != i and adj_matrix[i, node] == 0:
                    if opinions[node] == target_opinion:
                        same_tmp.append(node)
                    else:
                        diff_tmp.append(node)
        if len(same_tmp) > 0:
            candidates_same = np.array(same_tmp, dtype=np.int32)
        if len(diff_tmp) > 0:
            candidates_diff = np.array(diff_tmp, dtype=np.int32)

    else:
        return edges_to_remove, edges_to_add  # invalid search type

    # Choose pool (homophily)
    if np.random.rand() < homo_prob:
        candidates = candidates_same
    else:
        candidates = candidates_diff

    # fallback if empty
    if len(candidates) == 0:
        candidates = candidates_diff if len(candidates_same) == 0 else candidates_same
        if len(candidates) == 0:
            return edges_to_remove, edges_to_add

    # Select neighbor to remove
    if not force_replacement:
        mismatched = []
        for nb in current_neighbors:
            if opinions[nb] != target_opinion:
                mismatched.append(nb)
        if len(mismatched) == 0:
            return edges_to_remove, edges_to_add
        mismatched_arr = np.array(mismatched, dtype=np.int32)
        to_remove = mismatched_arr[np.random.randint(len(mismatched_arr))]
    else:
        to_remove = current_neighbors[np.random.randint(len(current_neighbors))]

    edges_to_remove[0, 0] = i
    edges_to_remove[0, 1] = to_remove
    edges_to_remove[1, 0] = to_remove
    edges_to_remove[1, 1] = i

    # Select edge to add
    new_node = candidates[np.random.randint(len(candidates))]
    if adj_matrix[i, new_node] == 0 and adj_matrix[new_node, i] == 0:
        edges_to_add[0, 0] = i
        edges_to_add[0, 1] = new_node
        edges_to_add[1, 0] = new_node
        edges_to_add[1, 1] = i

    return edges_to_remove, edges_to_add

@njit
def rewire_single_u_old(i, adj_matrix, opinions, search='global', force_replacement=True):
    """
    Undirected rewiring for node i.

    Parameters:
        i (int): Node to rewire.
        adj_matrix (np.ndarray): Adjacency matrix.
        opinions (np.ndarray): Array of opinions for each node.
        search (str): 'global' or 'local' search strategy.
        force_replacement (bool): Force rewiring even if opinions match.

    Returns:
        tuple: (edges_to_remove, edges_to_add)
    """
    max_edges = adj_matrix.shape[0] - 1  
    edges_to_remove = np.zeros((2, 2), dtype=np.int32)
    edges_to_add = np.zeros((2, 2), dtype=np.int32)
    edges_to_remove.fill(-1)
    edges_to_add.fill(-1)
    
    # Identify current neighbors
    current_neighbors = np.where(adj_matrix[i] == 1)[0]

    if len(current_neighbors) == 0 or len(current_neighbors) == max_edges:
        return edges_to_remove, edges_to_add
    
    target_opinion = opinions[i]
    
    # Check if all neighbors already share the same opinion
    if not force_replacement:
        all_same_opinion = True
        for neighbor in current_neighbors:
            if opinions[neighbor] != target_opinion:
                all_same_opinion = False
                break
        if all_same_opinion:
            return edges_to_remove, edges_to_add
    
    # Get non-neighbors
    all_nodes = np.arange(adj_matrix.shape[0])
    non_neighbors = np.setdiff1d(all_nodes, np.append(current_neighbors, i))
    
    # Find candidates for new connections
    if search == 'global':
        candidates = []
        for node in non_neighbors:
            if opinions[node] == target_opinion and node != i:
                candidates.append(node)
    elif search == 'local':
        candidates = []
        for neighbor in current_neighbors:
            neighbors_of_neighbor = np.where(adj_matrix[neighbor] == 1)[0]
            for node in neighbors_of_neighbor:
                if node != i and node not in current_neighbors and opinions[node] == target_opinion:
                    candidates.append(node)
    else:
        return edges_to_remove, edges_to_add  # Invalid search, do nothing
    
    if len(candidates) == 0:
        return edges_to_remove, edges_to_add
    
    # Select edge to remove
    if not force_replacement:
        mismatched_neighbors = []
        for neighbor in current_neighbors:
            if opinions[neighbor] != target_opinion:
                mismatched_neighbors.append(neighbor)
        
        if len(mismatched_neighbors) == 0:
            return edges_to_remove, edges_to_add
        
        to_remove = mismatched_neighbors[np.random.randint(len(mismatched_neighbors))]
    else:
        to_remove = current_neighbors[np.random.randint(len(current_neighbors))]
    
    edges_to_remove[0] = (i, to_remove)
    edges_to_remove[1] = (to_remove, i)
    
    # Select edge to add
    new_node = candidates[np.random.randint(len(candidates))]
    if adj_matrix[i, new_node] == 0 and adj_matrix[new_node, i] == 0:
        edges_to_add[0] = (i, new_node)
        edges_to_add[1] = (new_node, i)
    
    return edges_to_remove, edges_to_add

# ============================================================================
# OPINION UPDATE FUNCTIONS  
# ============================================================================

@njit
def update_opinions(opinions, adj_matrix):
    """
    Update opinions using the voter model dynamics.
    
    Randomly selects one node and updates its opinion based on a randomly 
    chosen neighbor. This implements the core voter model mechanism.

    Parameters:
        opinions (np.ndarray): Array of current opinion values for each node.
        adj_matrix (np.ndarray): Adjacency matrix of the graph.

    Returns:
        np.ndarray: Updated opinions array.
    """
    num_nodes = adj_matrix.shape[0]
    u = np.random.randint(0, num_nodes)
    neighbors = np.where(adj_matrix[u] == 1)[0]
    if len(neighbors) > 0:
        v = neighbors[np.random.randint(len(neighbors))]
        opinions[u] = opinions[v]
    return opinions

# ============================================================================
# HOMOPHILY CALCULATION FUNCTIONS
# ============================================================================

@njit
def calculate_individual_homophily(adj_matrix, opinions):
    """
    Calculate individual homophily for all nodes in the network.
    
    Homophily measures the tendency of nodes to connect to others with 
    similar opinions. Individual homophily is the fraction of a node's 
    neighbors that share the same opinion.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.

    Returns:
        np.ndarray: Individual homophily values for each node (range [0,1]).
    """
    num_agents = adj_matrix.shape[0]
    homophily = np.zeros(num_agents)
    
    for u in range(num_agents):
        neighbors = np.where(adj_matrix[u] == 1)[0]
        degree = len(neighbors)
        if degree == 0:
            homophily[u] = 1
            continue
        
        same_opinion = 0
        for v in neighbors:
            if opinions[u] == opinions[v]:
                same_opinion += 1
        homophily[u] = same_opinion / degree
    
    return homophily

@njit
def calculate_total_homophily(adj_matrix, opinions):
    """
    Calculate total homophily for the entire network.
    
    Total homophily is the fraction of all edges that connect nodes with 
    the same opinion. This provides a global measure of opinion segregation.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.

    Returns:
        float: Total homophily value (range [0,1]).
    """
    num_agents = adj_matrix.shape[0]
    total_same_opinion = 0
    total_links = 0
    
    for u in range(num_agents):
        neighbors = np.where(adj_matrix[u] == 1)[0]
        # Only count each edge once (u < v to avoid double counting)
        for v in neighbors:
            if u < v:  # This ensures we count each edge only once
                total_links += 1
                if opinions[u] == opinions[v]:
                    total_same_opinion += 1
    
    if total_links == 0:
        return 1.0  # If no links, define homophily as 1 by convention
    
    return total_same_opinion / total_links

@njit
def calculate_group_homophily(adj_matrix, opinions):
    """
    Calculate group-level homophily for different opinion groups.
    
    Group homophily measures the homophily within each opinion group 
    separately, providing insight into opinion-specific clustering patterns.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.

    Returns:
        tuple: (H0, H1, H_neg1) - homophily values for opinion groups 0, 1, and -1.
    """
    group_neg1_degree = 0
    group_neg1_same = 0
    group0_degree = 0
    group0_same = 0
    group1_degree = 0
    group1_same = 0
    
    num_agents = adj_matrix.shape[0]
    
    for u in range(num_agents):
        neighbors = np.where(adj_matrix[u] == 1)[0]
        degree = len(neighbors)
        same_opinion = 0
        
        for v in neighbors:
            if opinions[u] == opinions[v]:
                same_opinion += 1
        
        if opinions[u] == -1:
            group_neg1_degree += degree
            group_neg1_same += same_opinion
        elif opinions[u] == 0:
            group0_degree += degree
            group0_same += same_opinion
        else:  # opinion == 1
            group1_degree += degree
            group1_same += same_opinion
    
    H_neg1 = group_neg1_same / group_neg1_degree if group_neg1_degree > 0 else 0.0
    H0 = group0_same / group0_degree if group0_degree > 0 else 0.0
    H1 = group1_same / group1_degree if group1_degree > 0 else 0.0
    
    return H0, H1, H_neg1

# ============================================================================
# COMPONENT COUNTING FUNCTIONS
# ============================================================================

@njit
def count_opinion_homogeneous_components(adj_matrix, opinions, ignore=False, directed=False):
    """
    Count connected components where all nodes share the same opinion.
    
    This function identifies network components that have achieved perfect 
    local consensus, which is important for understanding polarization dynamics.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.
        ignore (bool): If True, only count components with at least 2 nodes.
        directed (bool): Whether to treat the graph as directed.
    
    Returns:
        int: Number of opinion-homogeneous connected components.
    """
    num_agents = adj_matrix.shape[0]
    visited = np.zeros(num_agents, dtype=np.int8)
    homogeneous_component_count = 0
    
    for node in range(num_agents):
        if not visited[node]:
            # Start DFS for this component
            stack = [node]
            visited[node] = 1
            component_nodes = [node]
            component_size = 1
            
            while stack:
                current = stack.pop()
                
                if directed:
                    # For directed graphs: weakly connected components
                    outgoing_neighbors = np.where(adj_matrix[current] == 1)[0]
                    incoming_neighbors = np.where(adj_matrix[:, current] == 1)[0]
                    
                    for neighbor in outgoing_neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_nodes.append(neighbor)
                            component_size += 1
                    
                    for neighbor in incoming_neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_nodes.append(neighbor)
                            component_size += 1
                else:
                    # For undirected graphs: standard DFS
                    neighbors = np.where(adj_matrix[current] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_nodes.append(neighbor)
                            component_size += 1
            
            # Check if all nodes in this component have the same opinion
            if component_size >= (2 if ignore else 1):
                component_opinion = opinions[component_nodes[0]]
                is_homogeneous = True
                
                for i in range(1, len(component_nodes)):
                    if opinions[component_nodes[i]] != component_opinion:
                        is_homogeneous = False
                        break
                
                if is_homogeneous:
                    homogeneous_component_count += 1
    
    return homogeneous_component_count

@njit
def count_connected_components(adj_matrix, ignore=False, directed=False):
    """
    Count connected components in the network.
    
    For undirected graphs: uses standard depth-first search.
    For directed graphs: counts weakly connected components.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        ignore (bool): If True, only count components with at least 2 nodes.
        directed (bool): Whether the graph is directed.
    
    Returns:
        int: Number of connected components.
    """
    num_agents = adj_matrix.shape[0]
    visited = np.zeros(num_agents, dtype=np.int8)
    component_count = 0
    
    for node in range(num_agents):
        if not visited[node]:
            stack = [node]
            visited[node] = 1
            component_size = 1
            
            while stack:
                current = stack.pop()
                
                if directed:
                    # For directed graphs: weakly connected components
                    outgoing_neighbors = np.where(adj_matrix[current] == 1)[0]
                    incoming_neighbors = np.where(adj_matrix[:, current] == 1)[0]
                    
                    for neighbor in outgoing_neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_size += 1
                    
                    for neighbor in incoming_neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_size += 1
                else:
                    # For undirected graphs: standard DFS
                    neighbors = np.where(adj_matrix[current] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
                            component_size += 1
            
            if not ignore or component_size >= 2:
                component_count += 1
    
    return component_count

@njit
def count_disconnected_components_old(adj_matrix, ignore=False):
    """
    Count disconnected components using depth-first search (legacy version).
    
    This is an older implementation maintained for backward compatibility.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        ignore (bool): If True, only count components with at least 2 nodes.
    
    Returns:
        int: Number of disconnected components.
    """
    num_agents = adj_matrix.shape[0]
    visited = np.zeros(num_agents, dtype=np.int8)
    component_count = 0
    
    for node in range(num_agents):
        if not visited[node]:
            stack = [node]
            visited[node] = 1
            component_size = 1  # Track size of current component
            
            while stack:
                current = stack.pop()
                neighbors = np.where(adj_matrix[current] == 1)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
                        component_size += 1
            
            # Only count if: not ignoring OR component has at least 2 nodes
            if not ignore and component_size >= 2:
                component_count += 1
    
    return component_count

@njit
def dfs_first_pass(adj_matrix, node, visited, stack):
    """
    First DFS pass for Kosaraju's algorithm to find strongly connected components.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the directed graph.
        node (int): Starting node for DFS.
        visited (np.ndarray): Array tracking visited nodes.
        stack (list): Stack to store finishing times.
    """
    visited[node] = 1
    for neighbor in range(adj_matrix.shape[0]):
        if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
            dfs_first_pass(adj_matrix, neighbor, visited, stack)
    stack.append(node)

@njit
def dfs_second_pass(adj_matrix_transpose, node, visited, component):
    """
    Second DFS pass for Kosaraju's algorithm on transposed graph.
    
    Parameters:
        adj_matrix_transpose (np.ndarray): Transposed adjacency matrix.
        node (int): Starting node for DFS.
        visited (np.ndarray): Array tracking visited nodes.
        component (list): List to store nodes in current component.
    """
    visited[node] = 1
    component.append(node)
    for neighbor in range(adj_matrix_transpose.shape[0]):
        if adj_matrix_transpose[node, neighbor] == 1 and not visited[neighbor]:
            dfs_second_pass(adj_matrix_transpose, neighbor, visited, component)

@njit
def count_strongly_connected_components(adj_matrix, ignore=False):
    """
    Count strongly connected components in a directed graph.
    
    Uses iterative implementation of Kosaraju's algorithm for efficiency
    and Numba compatibility.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the directed graph.
        ignore (bool): If True, only count components with at least 2 nodes.
    
    Returns:
        int: Number of strongly connected components.
    """
    num_agents = adj_matrix.shape[0]
    
    # First pass: iterative DFS to get finishing times
    visited = np.zeros(num_agents, dtype=np.int8)
    stack = []
    finish_times = []
    
    for node in range(num_agents):
        if not visited[node]:
            stack.append(node)
            visited[node] = 1
            
            while stack:
                current = stack[-1]
                found_unvisited = False
                
                for neighbor in range(num_agents):
                    if adj_matrix[current, neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
                        found_unvisited = True
                        break
                
                if not found_unvisited:
                    finish_times.append(stack.pop())
    
    # Create transpose of adjacency matrix
    adj_matrix_transpose = adj_matrix.T.copy()
    
    # Second pass: iterative DFS on transposed graph
    visited.fill(0)
    scc_count = 0
    
    # Process nodes in reverse order of finishing times
    for i in range(len(finish_times) - 1, -1, -1):
        node = finish_times[i]
        if not visited[node]:
            stack = [node]
            visited[node] = 1
            component_size = 1
            
            while stack:
                current = stack.pop()
                for neighbor in range(num_agents):
                    if adj_matrix_transpose[current, neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
                        component_size += 1
            
            if not ignore or component_size >= 2:
                scc_count += 1
    
    return scc_count

# ============================================================================
# CONVERGENCE ANALYSIS FUNCTIONS
# ============================================================================

@njit
def has_unconvergeable_components(adj_matrix, opinions):
    """
    Check for small disconnected components that cannot converge when phi=1.
    
    Identifies structural configurations that prevent convergence in pure 
    rewiring scenarios, which is important for understanding system dynamics.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.
    
    Returns:
        bool: True if unconvergeable components are found.
    """
    num_agents = adj_matrix.shape[0]
    visited = np.zeros(num_agents, dtype=np.bool_)
    
    for node in range(num_agents):
        if not visited[node]:
            # Start a new component
            component = []
            stack = [node]
            visited[node] = True
            component_size = 0
            
            # DFS to find component
            while stack:
                current = stack.pop()
                component.append(current)
                component_size += 1
                
                # We only care about small components (size <= 3 for efficiency)
                if component_size > 3:
                    break
                
                for neighbor in range(num_agents):
                    if adj_matrix[current, neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            
            # Check small components (size 2 or 3)
            if 2 <= component_size <= 3:
                component_opinions = opinions[np.array(component)]
                unique_opinions = np.unique(component_opinions)
                
                # If multiple opinions exist in component
                if len(unique_opinions) > 1:
                    # Check if any node has no same-opinion neighbor at distance 2
                    for i in range(component_size):
                        current_node = component[i]
                        current_opinion = opinions[current_node]
                        has_same_opinion_neighbor = False
                        
                        # Check immediate neighbors
                        for neighbor in range(num_agents):
                            if adj_matrix[current_node, neighbor] == 1 and opinions[neighbor] == current_opinion:
                                has_same_opinion_neighbor = True
                                break
                        
                        # If no same-opinion neighbor, check neighbors of neighbors
                        if not has_same_opinion_neighbor:
                            for neighbor in range(num_agents):
                                if adj_matrix[current_node, neighbor] == 1:
                                    for second_neighbor in range(num_agents):
                                        if adj_matrix[neighbor, second_neighbor] == 1 and opinions[second_neighbor] == current_opinion:
                                            has_same_opinion_neighbor = True
                                            break
                                    if has_same_opinion_neighbor:
                                        break
                        
                        # If a node has no same-opinion neighbor at distance <= 2, can't converge
                        if not has_same_opinion_neighbor:
                            return True
    
    return False

@njit
def has_unconvergeable_pairs(adj_matrix, opinions):
    """
    Check for isolated disagreeing pairs that cannot converge.
    
    Identifies pairs of connected nodes with different opinions that have 
    no shared neighbors, making convergence impossible in certain scenarios.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.
    
    Returns:
        bool: True if any unconvergeable pairs are found.
    """
    num_agents = adj_matrix.shape[0]
    
    for u in range(num_agents):
        for v in range(u+1, num_agents):  # Only check upper triangle
            if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                # Check if they share any neighbors
                has_shared_neighbor = False
                for w in range(num_agents):
                    if w == u or w == v:
                        continue
                    if adj_matrix[u, w] == 1 and adj_matrix[v, w] == 1:
                        has_shared_neighbor = True
                        break
                
                if not has_shared_neighbor:
                    return True  # Found isolated disagreeing pair
    return False

@njit
def check_convergence_phi1(adj_matrix, opinions):
    """
    Optimized convergence check for phi=1 (pure rewiring scenario).
    
    Uses a two-step approach: first checks for any disagreeing edges,
    then verifies if isolated disagreeing pairs exist that would prevent 
    convergence.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.
    
    Returns:
        bool: True if the system has converged.
    """
    num_agents = adj_matrix.shape[0]
    has_disagreeing_edges = False
    
    # First pass: check for any disagreeing edges
    for u in range(num_agents):
        for v in range(u+1, num_agents):  # Upper triangle only
            if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                has_disagreeing_edges = True
                # Early exit if we find an isolated disagreeing pair
                has_shared_neighbor = False
                for w in range(num_agents):
                    if w == u or w == v:
                        continue
                    if adj_matrix[u, w] == 1 and adj_matrix[v, w] == 1:
                        has_shared_neighbor = True
                        break
                
                if not has_shared_neighbor:
                    return False  # Immediate non-convergence
    
    return not has_disagreeing_edges  # True if no disagreeing edges exist

@njit
def check_convergence_phi1_old(adj_matrix, opinions):
    """
    Legacy convergence check for phi=1 scenario.
    
    Older implementation maintained for backward compatibility.
    Uses a different approach than the optimized version.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        opinions (np.ndarray): Array of opinion values for each node.
    
    Returns:
        bool: True if the system has converged.
    """
    # First check if any disagreeing edges exist
    num_agents = adj_matrix.shape[0]
    for u in range(num_agents):
        for v in range(num_agents):
            if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                # If disagreeing edges exist, check for unconvergeable components
                return not has_unconvergeable_components(adj_matrix, opinions)
    return True

@njit
def calculate_polarization_index(opinions):
    """
    Calculate polarization as distance from consensus.
    P = 0 at consensus, P = 1 at 50/50 split.
    """
    unique_opinions = np.unique(opinions)
    if len(unique_opinions) == 1:
        return 0.0
    
    count_1 = np.sum(opinions == unique_opinions[1])
    fraction_1 = count_1 / len(opinions)
    
    # 4*p*(1-p) gives maximum of 1 at p=0.5
    return 4.0 * fraction_1 * (1.0 - fraction_1)

@njit
def calculate_magnetization(opinions):
    """
    Calculate polarization as the absolute value of magnetization.
    P = |m| where m = (N_1 - N_0) / N
    P = 0 at balanced state (50/50), P = 1 at full consensus.
    """
    unique_opinions = np.unique(opinions)
    if len(unique_opinions) == 1:
        return 1.0  # Full consensus
    
    count_1 = np.sum(opinions == unique_opinions[1])
    count_0 = np.sum(opinions == unique_opinions[0])
    
    # Magnetization: |N_1 - N_0| / N
    magnetization = abs(count_1 - count_0) / len(opinions)
    
    return magnetization

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

@njit
def simulate_jit(adj_matrix, opinions_init, steps=100000, search='global', phi=0, 
                directed=False, force_replacement=True, track=True, ignore=False, n_steps=1000, homo_prob=1.0):
    """
    Main implementation of the Adaptive Voter Model (AVM) with comprehensive metrics.
    
    This is the primary simulation function that implements the full AVM dynamics
    with extensive tracking of network properties and opinion evolution. It combines
    voter model dynamics with homophily-driven rewiring to study opinion polarization.
    
    PERFORMANCE NOTES:
    - Use track=False for faster execution when only final results are needed
    - Increase n_steps to reduce measurement frequency for better performance  
    - Consider simulate_jit_fast or simulate_jit_minimal for large-scale runs
    
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        opinions_init (np.ndarray): Initial opinion values for each node.
        steps (int): Maximum number of Monte Carlo simulation steps.
        search (str): Rewiring strategy - 'global' or 'local' search for new connections.
        phi (float): Probability of rewiring vs. opinion update (range [0,1]).
        directed (bool): Whether the network is directed.
        force_replacement (bool): Whether to force rewiring even when satisfied.
        track (bool): Whether to track detailed time evolution metrics.
        ignore (bool): Whether to exclude isolated nodes from component measures.
        n_steps (int): Interval between metric measurements (affects resolution).
        homo_prob (float): Homophily probability for rewiring (range [0,1]).
        
    Returns:
        tuple: Comprehensive simulation results including:
            - m_diff (float): Final magnetization difference
            - adj_matrix (np.ndarray): Final network adjacency matrix  
            - opinions (np.ndarray): Final opinion configuration
            - convergence_time (float): Time to convergence (or -1 if no convergence)
            - clustering_history (np.ndarray): Evolution of clustering coefficient
            - individual_homophily_history (np.ndarray): Evolution of individual homophily
            - group_homophily_history (np.ndarray): Evolution of group homophily
            - total_homophily_history (np.ndarray): Evolution of total homophily
            - component_history (np.ndarray): Evolution of connected components
            - scc_history (np.ndarray): Evolution of strongly connected components
            - average_opinion_history (np.ndarray): Evolution of average opinion
            - homogeneous_component_history (np.ndarray): Evolution of opinion-homogeneous components
            - polarization_history (np.ndarray): Evolution of polarization index
    """
    num_agents = adj_matrix.shape[0]
    num_links = int(np.sum(adj_matrix))
    opinions = opinions_init.copy()  
    thres = np.random.rand(steps)
    
    # Calculate how many measurement points we'll have
    n_measurements = steps // n_steps + 1
    
    # Initialize all metric storage
    clustering_history = np.zeros(n_measurements) if track else np.zeros(1)
    individual_homophily_history = np.zeros(n_measurements) if track else np.zeros(1)
    group_homophily_history = np.zeros((n_measurements, 3)) if track else np.zeros((1, 3))
    total_homophily_history = np.zeros(n_measurements) if track else np.zeros(1)
    component_history = np.zeros(n_measurements) if track else np.zeros(1)
    scc_history = np.zeros(n_measurements) if track else np.zeros(1)
    average_opinion_history = np.zeros(n_measurements) if track else np.zeros(1)
    homogeneous_component_history = np.zeros(n_measurements) if track else np.zeros(1)
    polarization_history = np.zeros(n_measurements) if track else np.zeros(1)
    
    # Calculate initial metrics
    if track:
        clustering_history[0] = calculate_global_clustering_coefficient(adj_matrix)
        ind_h = calculate_individual_homophily(adj_matrix, opinions)
        tot_h = calculate_total_homophily(adj_matrix, opinions)
        H_neg1, H0, H1 = calculate_group_homophily(adj_matrix, opinions)
        
        # Calculate mean of individual homophily without using masked arrays
        valid_count = 0
        ind_h_sum = 0.0
        for i in range(len(ind_h)):
            if ind_h[i] >= 0:  # Only include valid values
                ind_h_sum += ind_h[i]
                valid_count += 1
        
        individual_homophily_history[0] = ind_h_sum / valid_count if valid_count > 0 else 0.0
        total_homophily_history[0] = tot_h
        group_homophily_history[0] = np.array([H_neg1, H0, H1])
        
        # Use appropriate component counting based on graph type
        if directed:
            component_history[0] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
            scc_history[0] = count_strongly_connected_components(adj_matrix, ignore=ignore)
        else:
            component_history[0] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
            scc_history[0] = component_history[0]  # For undirected, SCC = WCC
        
        # NEW: Track opinion-homogeneous components
        homogeneous_component_history[0] = count_opinion_homogeneous_components(
            adj_matrix, opinions, ignore=ignore, directed=directed
        )
        
        average_opinion_history[0] = np.mean(opinions)
        
        # NEW: Track polarization
        polarization_history[0] = calculate_magnetization(opinions)
    
    convergence_time = -1.0
    measurement_idx = 1
    
    for t in range(1, steps):
        # Model dynamics
        if thres[t] < 1 - phi:
            opinions = update_opinions(opinions, adj_matrix)
        else:
            edges_to_remove, edges_to_add = rewire(
                np.arange(num_agents), adj_matrix, opinions, 
                directed, search, force_replacement, homo_prob
            )
            # Update adjacency matrix
            for idx in range(edges_to_remove.shape[0]):
                u = edges_to_remove[idx, 0]
                v = edges_to_remove[idx, 1]
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 0
                    if not directed:
                        adj_matrix[v, u] = 0
            
            for idx in range(edges_to_add.shape[0]):
                u = edges_to_add[idx, 0]
                v = edges_to_add[idx, 1]
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 1
                    if not directed:
                        adj_matrix[v, u] = 1
        
        # Take measurements at intervals
        if t % n_steps == 0 or t == steps - 1:
            if track:
                clustering_history[measurement_idx] = calculate_global_clustering_coefficient(adj_matrix)
                ind_h = calculate_individual_homophily(adj_matrix, opinions)
                H_neg1, H0, H1 = calculate_group_homophily(adj_matrix, opinions)
                tot_h = calculate_total_homophily(adj_matrix, opinions)
                
                # Calculate mean without masked arrays
                valid_count = 0
                ind_h_sum = 0.0
                for i in range(len(ind_h)):
                    if ind_h[i] >= 0:
                        ind_h_sum += ind_h[i]
                        valid_count += 1
                
                individual_homophily_history[measurement_idx] = ind_h_sum / valid_count if valid_count > 0 else 0.0
                total_homophily_history[measurement_idx] = tot_h
                group_homophily_history[measurement_idx] = np.array([H_neg1, H0, H1])
                
                # Use appropriate component counting based on graph type
                if directed:
                    component_history[measurement_idx] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
                    scc_history[measurement_idx] = count_strongly_connected_components(adj_matrix, ignore=ignore)
                else:
                    component_history[measurement_idx] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
                    scc_history[measurement_idx] = component_history[measurement_idx]
                
                # NEW: Track opinion-homogeneous components
                homogeneous_component_history[measurement_idx] = count_opinion_homogeneous_components(
                    adj_matrix, opinions, ignore=ignore, directed=directed
                )
                
                average_opinion_history[measurement_idx] = np.mean(opinions)
                
                # NEW: Track polarization
                polarization_history[measurement_idx] = calculate_magnetization(opinions)
            
            measurement_idx += 1
        
        # Check for convergence
        if int(phi) == 1 and search == 'local':
            converged = check_convergence_phi1(adj_matrix, opinions)
        else:
            converged = True
            for u in range(num_agents):
                for v in range(num_agents):
                    if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                        converged = False
                        break
                if not converged:
                    break
        
        if converged:
            convergence_time = float(t) / num_agents
            # Take one final measurement to capture the converged state
            if track:
                clustering_history[measurement_idx] = calculate_global_clustering_coefficient(adj_matrix)
                ind_h = calculate_individual_homophily(adj_matrix, opinions)
                
                # Calculate mean without masked arrays
                valid_count = 0
                ind_h_sum = 0.0
                for i in range(len(ind_h)):
                    if ind_h[i] >= 0:
                        ind_h_sum += ind_h[i]
                        valid_count += 1
                
                individual_homophily_history[measurement_idx] = ind_h_sum / valid_count if valid_count > 0 else 0.0
                total_homophily_history[measurement_idx] = calculate_total_homophily(adj_matrix, opinions)
                H_neg1, H0, H1 = calculate_group_homophily(adj_matrix, opinions)
                group_homophily_history[measurement_idx] = np.array([H_neg1, H0, H1])
                
                # Use appropriate component counting based on graph type
                if directed:
                    component_history[measurement_idx] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
                    scc_history[measurement_idx] = count_strongly_connected_components(adj_matrix, ignore=ignore)
                else:
                    component_history[measurement_idx] = count_connected_components(adj_matrix, ignore=ignore, directed=directed)
                    scc_history[measurement_idx] = component_history[measurement_idx]
                
                # NEW: Track opinion-homogeneous components
                homogeneous_component_history[measurement_idx] = count_opinion_homogeneous_components(
                    adj_matrix, opinions, ignore=ignore, directed=directed
                )
                
                average_opinion_history[measurement_idx] = np.mean(opinions)
                
                # NEW: Track polarization
                polarization_history[measurement_idx] = calculate_magnetization(opinions)
                
                measurement_idx += 1
            
            # Pad the remaining history with the true final state
            if track:
                for idx in range(measurement_idx, n_measurements):
                    clustering_history[idx] = clustering_history[measurement_idx-1]
                    individual_homophily_history[idx] = individual_homophily_history[measurement_idx-1]
                    total_homophily_history[idx] = total_homophily_history[measurement_idx-1]
                    group_homophily_history[idx] = group_homophily_history[measurement_idx-1]
                    component_history[idx] = component_history[measurement_idx-1]
                    scc_history[idx] = scc_history[measurement_idx-1]
                    homogeneous_component_history[idx] = homogeneous_component_history[measurement_idx-1]
                    average_opinion_history[idx] = average_opinion_history[measurement_idx-1]
                    polarization_history[idx] = polarization_history[measurement_idx-1]
            break
    
    if convergence_time == -1.0:
        # System did not converge - removed print statement for Numba compatibility
        # Final padding if not converged
        if track:
            for idx in range(measurement_idx, n_measurements):
                clustering_history[idx] = clustering_history[measurement_idx-1]
                individual_homophily_history[idx] = individual_homophily_history[measurement_idx-1]
                total_homophily_history[idx] = total_homophily_history[measurement_idx-1]
                group_homophily_history[idx] = group_homophily_history[measurement_idx-1]
                component_history[idx] = component_history[measurement_idx-1]
                scc_history[idx] = scc_history[measurement_idx-1]
                homogeneous_component_history[idx] = homogeneous_component_history[measurement_idx-1]
                average_opinion_history[idx] = average_opinion_history[measurement_idx-1]
                polarization_history[idx] = polarization_history[measurement_idx-1]
    
    # Calculate final magnetization
    M_neg1neg1 = 0
    M_00 = 0
    M_11 = 0
    for u in range(num_agents):
        for v in range(num_agents):
            if adj_matrix[u, v] == 1:
                if opinions[u] == opinions[v] == -1:
                    M_neg1neg1 += 1
                elif opinions[u] == opinions[v] == 0:
                    M_00 += 1
                elif opinions[u] == opinions[v] == 1:
                    M_11 += 1
    
    m_diff = float(M_11 - M_neg1neg1) / num_links if num_links > 0 else 0.0
    
    return (
        m_diff, 
        adj_matrix, 
        opinions,  
        convergence_time, 
        clustering_history,
        individual_homophily_history,
        group_homophily_history,
        total_homophily_history,
        component_history,
        scc_history,
        average_opinion_history,
        homogeneous_component_history,  
        polarization_history  
    )

@njit
def simulate_jit_fast(adj_matrix, opinions_init, steps=100000, search='global', phi=0, 
                     directed=False, force_replacement=True, track_lite=True, n_steps=5000):
    """
    Performance-optimized AVM simulation with reduced metric calculations.
    
    This version prioritizes speed over comprehensive tracking, making it suitable
    for large-scale parameter sweeps or when only essential metrics are needed.
    
    Key optimizations:
    - Less frequent metric calculations (default n_steps=5000)
    - Simplified convergence checking with early termination
    - Lightweight tracking mode for essential metrics only
    - Pre-allocated arrays to reduce memory overhead
    - Optimized neighbor lookup caching
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        opinions_init (np.ndarray): Initial opinion values for each node.
        steps (int): Maximum number of simulation steps.
        search (str): Rewiring strategy - 'global' or 'local'.
        phi (float): Probability of rewiring vs. opinion update.
        directed (bool): Whether the network is directed.
        force_replacement (bool): Whether to force rewiring even when satisfied.
        track_lite (bool): If True, only track essential metrics for performance.
        n_steps (int): Measurement interval (larger values = faster execution).
        
    Returns:
        tuple: Essential simulation results:
            - m_diff (float): Final magnetization difference
            - adj_matrix (np.ndarray): Final network state
            - opinions (np.ndarray): Final opinion configuration
            - convergence_time (float): Time to convergence
            - individual_homophily_history (np.ndarray): Homophily evolution
            - average_opinion_history (np.ndarray): Opinion evolution
    """
    num_agents = adj_matrix.shape[0]
    num_links = int(np.sum(adj_matrix))
    opinions = opinions_init.copy()  
    thres = np.random.rand(steps)
    
    # Calculate measurement points - fewer for better performance
    n_measurements = steps // n_steps + 1
    
    # Initialize only essential metric storage for performance
    individual_homophily_history = np.zeros(n_measurements) if track_lite else np.zeros(1)
    average_opinion_history = np.zeros(n_measurements) if track_lite else np.zeros(1)
    
    # Pre-allocate temporary arrays for repeated use
    temp_homophily = np.zeros(num_agents)
    
    # Calculate initial metrics (essential only)
    if track_lite:
        # Fast individual homophily calculation
        for u in range(num_agents):
            degree = 0
            same_opinion = 0
            for v in range(num_agents):
                if adj_matrix[u, v] == 1:
                    degree += 1
                    if opinions[u] == opinions[v]:
                        same_opinion += 1
        
            temp_homophily[u] = same_opinion / degree if degree > 0 else 1.0
        
        individual_homophily_history[0] = np.mean(temp_homophily)
        average_opinion_history[0] = np.mean(opinions)
    
    convergence_time = -1
    measurement_idx = 1
    
    # Cache for faster convergence checking
    last_convergence_check = 0
    convergence_check_interval = min(1000, n_steps // 2)  # Check less frequently
    
    for t in range(1, steps):
        # Model dynamics
        if thres[t] < 1 - phi:
            opinions = update_opinions(opinions, adj_matrix)
        else:
            edges_to_remove, edges_to_add = rewire(
                np.arange(num_agents), adj_matrix, opinions, 
                directed, search, force_replacement
            )
            # Update adjacency matrix
            for u, v in edges_to_remove:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 0
                    if not directed:
                        adj_matrix[v, u] = 0
            for u, v in edges_to_add:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 1
                    if not directed:
                        adj_matrix[v, u] = 1
        
        # Take measurements at intervals (less frequent for performance)
        if t % n_steps == 0 or t == steps - 1:
            if track_lite:
                # Fast homophily calculation
                for u in range(num_agents):
                    degree = 0
                    same_opinion = 0
                    for v in range(num_agents):
                        if adj_matrix[u, v] == 1:
                            degree += 1
                            if opinions[u] == opinions[v]:
                                same_opinion += 1
                    temp_homophily[u] = same_opinion / degree if degree > 0 else 1.0
                
                individual_homophily_history[measurement_idx] = np.mean(temp_homophily)
                average_opinion_history[measurement_idx] = np.mean(opinions)
            
            measurement_idx += 1
        
        # Optimized convergence check - less frequent and with early termination
        if t - last_convergence_check >= convergence_check_interval:
            converged = True
            disagreeing_edges = 0
            
            # Quick scan for disagreeing edges with early termination
            for u in range(num_agents):
                for v in range(u + 1, num_agents):  # Only upper triangle
                    if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                        disagreeing_edges += 1
                        converged = False
                        if disagreeing_edges > 10:  # Early termination if too many disagreements
                            break
                if not converged and disagreeing_edges > 10:
                    break
            
            last_convergence_check = t
            
            if converged:
                convergence_time = t / num_agents
                # Final measurement
                if track_lite:
                    for u in range(num_agents):
                        degree = 0
                        same_opinion = 0
                        for v in range(num_agents):
                            if adj_matrix[u, v] == 1:
                                degree += 1
                                if opinions[u] == opinions[v]:
                                    same_opinion += 1
                        temp_homophily[u] = same_opinion / degree if degree > 0 else 1.0
                    
                    individual_homophily_history[measurement_idx] = np.mean(temp_homophily)
                    average_opinion_history[measurement_idx] = np.mean(opinions)
                    measurement_idx += 1
                
                # Pad remaining history
                if track_lite:
                    individual_homophily_history[measurement_idx:] = individual_homophily_history[measurement_idx-1]
                    average_opinion_history[measurement_idx:] = average_opinion_history[measurement_idx-1]
                break
    
    if convergence_time == -1:
        print("System did not converge!")
        if track_lite:
            individual_homophily_history[measurement_idx:] = individual_homophily_history[measurement_idx-1]
            average_opinion_history[measurement_idx:] = average_opinion_history[measurement_idx-1]
    
    # Calculate final magnetization (simplified)
    same_opinion_links = 0
    for u in range(num_agents):
        for v in range(u + 1, num_agents):  # Only count each edge once
            if adj_matrix[u, v] == 1 and opinions[u] == opinions[v]:
                same_opinion_links += 1
    
    m_diff = same_opinion_links / num_links if num_links > 0 else 0.0
    
    return (
        m_diff, 
        adj_matrix, 
        opinions,  
        convergence_time, 
        individual_homophily_history,
        average_opinion_history
    )

@njit
def simulate_jit_minimal(adj_matrix, opinions_init, steps=100000, search='global', phi=0, 
                        directed=False, force_replacement=True):
    """
    Ultra-fast AVM simulation with minimal tracking for maximum performance.
    
    This version eliminates all time evolution tracking and focuses purely on 
    speed, making it ideal for large-scale parameter exploration where only 
    final states matter. Can be 5-10x faster than the full simulation.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        opinions_init (np.ndarray): Initial opinion values for each node.
        steps (int): Maximum number of simulation steps.
        search (str): Rewiring strategy - 'global' or 'local'.
        phi (float): Probability of rewiring vs. opinion update.
        directed (bool): Whether the network is directed.
        force_replacement (bool): Whether to force rewiring even when satisfied.
    
    Returns:
        tuple: Minimal results for maximum speed:
            - magnetization (float): Final magnetization value
            - adj_matrix (np.ndarray): Final network state
            - opinions (np.ndarray): Final opinion configuration  
            - convergence_time (float): Time to convergence (or -1)
    """
    num_agents = adj_matrix.shape[0]
    opinions = opinions_init.copy()  
    thres = np.random.rand(steps)
    
    convergence_time = -1
    last_check = 0
    check_interval = 5000  # Check convergence less frequently
    
    for t in range(1, steps):
        # Model dynamics
        if thres[t] < 1 - phi:
            opinions = update_opinions(opinions, adj_matrix)
        else:
            edges_to_remove, edges_to_add = rewire(
                np.arange(num_agents), adj_matrix, opinions, 
                directed, search, force_replacement
            )
            # Update adjacency matrix
            for u, v in edges_to_remove:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 0
                    if not directed:
                        adj_matrix[v, u] = 0
            for u, v in edges_to_add:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 1
                    if not directed:
                        adj_matrix[v, u] = 1
        
        # Minimal convergence checking
        if t - last_check >= check_interval:
            converged = True
            # Quick scan with very early termination
            for u in range(num_agents):
                for v in range(u + 1, min(u + 50, num_agents)):  # Check only subset
                    if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                        converged = False
                        break
                if not converged:
                    break
            
            if converged:
                # Full convergence check only when quick scan passes
                converged = True
                for u in range(num_agents):
                    for v in range(u + 1, num_agents):
                        if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                            converged = False
                            break
                    if not converged:
                        break
                
                if converged:
                    convergence_time = t / num_agents
                    break
            
            last_check = t
    
    # Calculate final magnetization
    num_links = int(np.sum(adj_matrix)) // 2 if not directed else int(np.sum(adj_matrix))
    same_opinion_links = 0
    
    if not directed:
        for u in range(num_agents):
            for v in range(u + 1, num_agents):
                if adj_matrix[u, v] == 1 and opinions[u] == opinions[v]:
                    same_opinion_links += 1
    else:
        for u in range(num_agents):
            for v in range(num_agents):
                if adj_matrix[u, v] == 1 and opinions[u] == opinions[v]:
                    same_opinion_links += 1
    
    magnetization = same_opinion_links / num_links if num_links > 0 else 0.0
    
    return magnetization, adj_matrix, opinions, convergence_time


@njit
def simulate_jit_minimal(adj_matrix, opinions_init, steps=100000, search='global', phi=0, 
                        directed=False, force_replacement=True):
    """
    Ultra-fast AVM simulation with minimal tracking for maximum performance.
    
    This version eliminates all time evolution tracking and focuses purely on 
    speed, making it ideal for large-scale parameter exploration where only 
    final states matter. Can be 5-10x faster than the full simulation.
    
    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        opinions_init (np.ndarray): Initial opinion values for each node.
        steps (int): Maximum number of simulation steps.
        search (str): Rewiring strategy - 'global' or 'local'.
        phi (float): Probability of rewiring vs. opinion update.
        directed (bool): Whether the network is directed.
        force_replacement (bool): Whether to force rewiring even when satisfied.
    
    Returns:
        tuple: Minimal results for maximum speed:
            - magnetization (float): Final magnetization value
            - adj_matrix (np.ndarray): Final network state
            - opinions (np.ndarray): Final opinion configuration  
            - convergence_time (float): Time to convergence (or -1)
    """
    num_agents = adj_matrix.shape[0]
    opinions = opinions_init.copy()  
    thres = np.random.rand(steps)
    
    convergence_time = -1
    last_check = 0
    check_interval = 5000  # Check convergence less frequently
    
    for t in range(1, steps):
        # Model dynamics
        if thres[t] < 1 - phi:
            opinions = update_opinions(opinions, adj_matrix)
        else:
            edges_to_remove, edges_to_add = rewire(
                np.arange(num_agents), adj_matrix, opinions, 
                directed, search, force_replacement
            )
            # Update adjacency matrix
            for u, v in edges_to_remove:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 0
                    if not directed:
                        adj_matrix[v, u] = 0
            for u, v in edges_to_add:
                if u != -1 and v != -1:
                    adj_matrix[u, v] = 1
                    if not directed:
                        adj_matrix[v, u] = 1
        
        # Minimal convergence checking
        if t - last_check >= check_interval:
            converged = True
            # Quick scan with very early termination
            for u in range(num_agents):
                for v in range(u + 1, min(u + 50, num_agents)):  # Check only subset
                    if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                        converged = False
                        break
                if not converged:
                    break
            
            if converged:
                # Full convergence check only when quick scan passes
                converged = True
                for u in range(num_agents):
                    for v in range(u + 1, num_agents):
                        if adj_matrix[u, v] == 1 and opinions[u] != opinions[v]:
                            converged = False
                            break
                    if not converged:
                        break
                
                if converged:
                    convergence_time = t / num_agents
                    break
            
            last_check = t
    
    # Calculate final magnetization
    num_links = int(np.sum(adj_matrix)) // 2 if not directed else int(np.sum(adj_matrix))
    same_opinion_links = 0
    
    if not directed:
        for u in range(num_agents):
            for v in range(u + 1, num_agents):
                if adj_matrix[u, v] == 1 and opinions[u] == opinions[v]:
                    same_opinion_links += 1
    else:
        for u in range(num_agents):
            for v in range(num_agents):
                if adj_matrix[u, v] == 1 and opinions[u] == opinions[v]:
                    same_opinion_links += 1
    
    magnetization = same_opinion_links / num_links if num_links > 0 else 0.0
    
    return magnetization, adj_matrix, opinions, convergence_time

# ============================================================================
# OBJECT-ORIENTED INTERFACE CLASS
# ============================================================================

class Rewiring_Model:
    def __init__(self, graph, opinion_init=2, opinion_bounds=(-1, 1), custom_opinions=None, change=False, n1_0=0.5):
        """
        Initialize the model with basic settings.
        
        graph: Input graph representing the social network.
        opinion_init: Number of different opinions 
        opinion_bounds: Bounds for the random opinion distribution (min, max).
        custom_opinions: Custom opinions for initialization if using 'survey' method.
        change: Randomly shuffle opinions around the network
        n1_0: Initial fraction of opinions in state 1
        """
        self.graph = deepcopy(graph)
        if not self.graph.is_directed():
            self.graph = self.graph.to_directed()
        
        self.N = len(self.graph.nodes())
        self.min_opinion, self.max_opinion = opinion_bounds
        self.custom_opinions = custom_opinions
        self.adj_matrix = nx.to_numpy_array(self.graph)
        # Initialize opinions based on the specified method
        self.change = change
        self.opinions = self.initialize_opinions_new(opinion_init, [n1_0, 1 - n1_0])

        # Keep track of initial opinions and history for analysis
        self.opinions_init = deepcopy(self.opinions)
        self.opinion_history = []
        self.influencers = None
        self.external_influencers = None
        self.layout = nx.kamada_kawai_layout(self.graph)
    
    def initialize_opinions_new(self, num_opinions, fractions=None):
        """
        Initialize opinions drawn from an equally spaced linspace within the opinion bounds.

        num_opinions: Number of distinct opinions (int)
        fractions: List of fractions for each opinion state (optional, must sum to 1)
        return: Initial opinion array
        """
        if not isinstance(num_opinions, int) or num_opinions <= 0:
            raise ValueError("Number of opinions must be a positive integer.")

        if fractions is not None:
            if len(fractions) != num_opinions:
                raise ValueError("Length of fractions must match the number of opinions.")
            if not np.isclose(sum(fractions), 1):
                raise ValueError("Fractions must sum to 1.")
        else:
            fractions = [1 / num_opinions] * num_opinions

        # Generate equally spaced opinion values
        opinion_values = np.linspace(self.min_opinion, self.max_opinion, num_opinions)
        #print("Initial opinions new op: ", fractions)
        # Calculate the number of agents per opinion based on fractions
        opinions = []
        for opinion, fraction in zip(opinion_values, fractions):
            count = int(fraction * self.N)
            opinions.extend([opinion] * count)

        # Adjust for rounding errors to ensure total count equals N
        while len(opinions) < self.N:
            opinions.append(np.random.choice(opinion_values))
        opinions = opinions[:self.N]

        # Shuffle opinions to randomize their initial placement
        np.random.shuffle(opinions)

        # Convert to numpy array for compatibility
        opinions = np.array(opinions)

        return opinions

    def add_external_influencers(self, n_external, external_opinions=None):
        """
        Select n existing nodes as external influencers, remove all in-links, and
        add only out-links to half of the network nodes, excluding self-connections.
        
        n_external: Number of external influencers.
        external_opinions: List or array of opinion values to assign to the external influencers.
        """
        # Randomly select n_external nodes from the graph as influencers
        self.external_influencers = np.random.choice(self.graph.nodes(), n_external, replace=False)

        # Set the opinions of the external influencers
        if external_opinions:
            if len(external_opinions) != n_external:
                raise ValueError("Number of external_opinions must match n_external.")
        
            for idx, influencer in enumerate(self.external_influencers):
                # Change opinions of external influencers to input values
                self.opinions_init[influencer] = external_opinions[idx]

        else:
            for idx, influencer in enumerate(self.external_influencers):
                # Change opinions to upper bound if no values are given
                self.opinions_init[influencer] = self.max_opinion
        
        # Define the target subset of nodes for out-links (half the network)
        half_network = list(self.graph.nodes())[:self.N // 2]

        for influencer in self.external_influencers:
            # Remove all incoming edges to this influencer node to ensure it has only out-links
            in_edges = list(self.graph.in_edges(influencer))
            self.graph.remove_edges_from(in_edges)

            # Create out-links from influencer to half of the network nodes, excluding self-connections
            # Ensure that no external influencer is linked to another external influencer
            out_links = [(influencer, node) for node in half_network if node != influencer and node not in self.external_influencers]
            
            # If half the network is smaller than needed, we might need to adjust the out-links selection
            if len(out_links) < (len(half_network) // 2):  # Ensure we connect to half of the half_network
                out_links = out_links[:len(out_links) // 2]  # Adjust to only half of available out-links

            self.graph.add_edges_from(out_links)

    def update_opinions(self) -> None:
        """Update random node to state of random neighbor
        """
        #new_opinions = deepcopy(self.opinions)
        #new_opinions = self.opinions.copy()  

        i = np.random.choice(self.influencers)
        neighbors = list(self.graph.predecessors(i))  # Get in-links for node i
        if len(neighbors) > 0:
            j = np.random.choice(neighbors)
            #new_opinions[i] = self.opinions[j]  
            self.opinions[i] = self.opinions[j]              
        #j = np.random.choice(self.graph.nodes) 
        #new_opinions[i] = self.opinions[j]

        #return new_opinions     

    def rewire(self, search='global', directed=True) -> None:
        """
       Rewire influencers based on opinion matching in a discrete system.
        
        Parameters:
            search (str): Search space for rewiring, either 'global' or 'neighbor'.
            wire (str): Rewiring type, either 'all' or 'single'.
            target (str): Rewiring target type, either 'individual' or 'majority'
            directed (bool): Directed rewiring of True, otherwise undirected rewiring
        
        Raises:
            TypeError: If `self.influencers` is not a numpy array.
        """
        edges_to_remove = []
        edges_to_add = []

        if not isinstance(self.influencers, np.ndarray):
            raise TypeError("self.influencers must be an array.")
        
        # Choose node for rewiring
        i = np.random.choice(self.influencers) 
        if directed == True:
            edges_to_remove, edges_to_add = self._rewire_single_d(i, search)
        else:
            edges_to_remove, edges_to_add = self._rewire_single_u(i, search)
        
        # Apply rewiring changes
        self.graph.remove_edges_from(edges_to_remove)
        self.graph.add_edges_from(edges_to_add)
    
    def _rewire_single_u(self, i, search='global') -> tuple[tuple, tuple]:
        """
        Determine rewiring changes for a node in a discrete opinion system,
        treating the graph as undirected to simplify the logic.

        Parameters:
            i (int): The target node for rewiring.
            search (str): 'global' for all non-neighbors or 'local' for neighbors-of-neighbors.

        Returns:
            tuple: A tuple (edges_to_remove, edges_to_add) where each element is a tuple of edges.

        Raises:
            ValueError: If `search` is neither `'global'` or `'local'`
        """
        edges_to_remove = []
        edges_to_add = []

        # Identify current neighbors of node `i` (treating the graph as undirected)
        current_neighbors = set(self.graph.neighbors(i)).union(
            {node for node in self.graph.predecessors(i) if node != i}
        )

        # If the node is isolated or fully connected, no rewiring is needed
        if len(current_neighbors) == 0 or len(current_neighbors) == len(self.graph) - 1:
            return edges_to_remove, edges_to_add

        # Get opinions of all nodes
        opinions = self.opinions

        # Determine the target opinion
        target_opinion = opinions[i]

        # Check if all neighbors already share the same opinion
        if all(opinions[neighbor] == target_opinion for neighbor in current_neighbors) and not self.force_replacement:
            return [], []

        # Identify candidates for new edges before removing any edges
        all_nodes = set(self.graph.nodes())
        non_neighbors = all_nodes - current_neighbors - {i}

        if search == 'global':
            candidates = [node for node in non_neighbors if opinions[node] == target_opinion and node != i]
        elif search == 'local':
            neighbors_of_neighbors = set()
            for neighbor in current_neighbors:
                neighbors_of_neighbors.update(self.graph.predecessors(neighbor))
            neighbors_of_neighbors -= current_neighbors  # Exclude current neighbors
            neighbors_of_neighbors -= {i}  # Exclude self
            candidates = [node for node in neighbors_of_neighbors if opinions[node] == target_opinion and node != i]
        else:
            raise ValueError("Invalid search parameter. Use 'global' or 'local'.")

        # If no candidates are available, skip rewiring
        if not candidates:
            return [], []

        # Identify an edge to remove based on opinion mismatch or random removal
        if not self.force_replacement:
            # Remove a random neighbor with a mismatched opinion
            mismatched_neighbors = [neighbor for neighbor in current_neighbors if opinions[neighbor] != target_opinion]
            if not mismatched_neighbors:
                return [], []  # No mismatched neighbors, no rewiring
            edge_to_remove = (i, mismatched_neighbors[np.random.randint(len(mismatched_neighbors))])
        else:
            # Remove a random neighbor (regardless of opinion)
            edge_to_remove = (i, list(current_neighbors)[np.random.randint(len(current_neighbors))])

        edges_to_remove.append(edge_to_remove)
        edges_to_remove.append((edge_to_remove[1], edge_to_remove[0]))  # Ensure undirected removal

        # Add a new edge to a valid candidate, ensuring no self-loops or multi-edges
        new_node = np.random.choice(candidates)
        # Ensure the new edge does not already exist
        if not self.graph.has_edge(i, new_node) and not self.graph.has_edge(new_node, i):
            edges_to_add.append((i, new_node))
            edges_to_add.append((new_node, i))  # Ensure undirected addition
        else:
            raise ValueError(f"Trying to create multiedge for edge {i}, check edge selection for {self.directed_val}!")

        #print(f'Removed edges of node {i}: {edges_to_remove}')
        #print(f'Added edges of node {i}: {edges_to_add}')

        return edges_to_remove, edges_to_add

    def _rewire_single_d(self, i, search='global') -> tuple[tuple, tuple]:
        """
        Determine rewiring changes for a node in a discrete opinion system.

        Parameters:
            i (int): The target node for rewiring.
            search (str): 'global' for all non-neighbors or 'neighbor' for neighbors-of-neighbors.

        Returns:
            tuple: A tuple (edges_to_remove, edges_to_add) of edges to add and remove.

        Raises:
            ValueError: If `search` is neither `'global'` or `'local'`
        """
        edges_to_remove = []
        edges_to_add = []

        # Identify current incoming edges to node `i`
        current_in_edges = list(self.graph.in_edges(i))
        if len(current_in_edges) == 0 or len(current_in_edges) == self.N - 1:
            return [], []  # No rewiring needed

        # Get opinions of all nodes
        opinions = self.opinions

        # Determine the target opinion
        target_opinion = opinions[i]

        # Check if all in-edges already share the same opinion
        if all(opinions[edge[0]] == target_opinion for edge in current_in_edges) and not self.force_replacement:
            return [], []  # No rewiring needed

        # Identify mismatched edges
        mismatched_edges = [edge for edge in current_in_edges if opinions[edge[0]] != target_opinion]
        if not mismatched_edges and not self.force_replacement: # No mismatched edges and no forced rewiring, then no changes
            return [], []  

        # Select an edge to remove
        if not self.force_replacement and mismatched_edges:  # Remove a mismatched in-edge if no forced replacement
            edge_to_remove = mismatched_edges[np.random.randint(len(mismatched_edges))]
            edges_to_remove.append(edge_to_remove)
        elif self.force_replacement and current_in_edges:
            edge_to_remove = current_in_edges[np.random.randint(len(current_in_edges))]
            edges_to_remove.append(edge_to_remove)
        else:
            print(f"No current or mismatched edges for node {i}!")

        # Determine candidates for a new edge
        all_nodes = set(self.graph.nodes())
        current_sources = {src for src, _ in current_in_edges}

        if search == 'global': # Choose any node not already connected to `i` and with the target opinion
            candidates = [node for node in all_nodes if opinions[node] == target_opinion and node not in current_sources and node != i]
        elif search == 'local': # Consider nodes two hops away of the target opinion
            #neighbors = set(self.graph.neighbors(i))
            neighbors = set(self.graph.in_edges(i))
            nb_of_nb = set()
            for neighbor in neighbors:
                for src, _ in self.graph.in_edges(neighbor):
                    if src != i and src not in neighbors:
                        nb_of_nb.add(src)

            candidates = [node for node in nb_of_nb if opinions[node] == target_opinion and node not in current_sources and node != i]
        else:
            raise ValueError("Invalid search parameter. Use 'global' or 'local'.")

        # If no candidates are available, skip rewiring
        if not candidates:
            return [], []

        # Add a new edge to a valid candidate
        new_node = np.random.choice(candidates)
        if not self.graph.has_edge(new_node, i):
            edges_to_add.append((new_node, i))
        else:
            raise ValueError(f"Trying to create multiedge for edge {i}, check edge selection for {self.directed_val}!")
        
        #print(f'Removed edge of node {i}: {edges_to_remove}')
        #print(f'Added edge of node {i}: {edges_to_add}')

        return edges_to_remove, edges_to_add

    def count_influencers(self, inf_th=0.5) -> int:
        """Count the number of influencers in the graph.

        inf_th: Fraction of nodes to be labeled as influencer

        Return:
            num influencer (int): Number of influencers
        """
        threshold = self.N * inf_th
        out_degrees = dict(self.graph.out_degree()) 
        return sum(1 for _, degree in out_degrees.items() if degree > threshold)
    

    def simulate_slim(self, m_int=0, steps=100000, search='global', draw_graph=False, inf_th=0.6, phi=0, 
                      directed=True, force_replacement=True, f_z=0) -> tuple[list, list, list, list]:
        """Run the simulation, calculate convergence time as updates per vertex to reach equilibrium.
        
        Parameters:

            m_int (int): Number of influencers.
            steps (int): Maximum number of simulation steps.
            search (str): Option to let the rewiring choice be the whole graph ('global') or neigh-of-neigh ('local')
            draw_graph (bool): Draw initial and final graph if True
            phi (float): Probability of rewiring instead of opinion update.
            directed (bool): Whether the graph is directed.
            force_replacement (float): Randomly delete link (True) or delete opposite links (False)
            f_z (float): Fraction of nodes that do not change their opinion, always opinion 1

        Returns:

            results (tuple): Tuple of opinion history, average history, convergence time and component sizes

        Raises:

            ValueError: If the number of links before and after rewiring changes
        """
        #if m_int is None:
        #    self.m_int = self.N
        #else:
        #    self.m_int = m_int
        if m_int > 0:
            self.influencers = np.array(list(range(m_int)))

        self.force_replacement = force_replacement
        self.convergence_time = None

        # Zealot setup
        num_zealots = int(f_z * self.N)
        # Find indices of nodes already in state 1
        state_1_indices = np.where(self.opinions == 1)[0]
        # Randomly select zealots from nodes in state 1
        zealot_indices = np.random.choice(state_1_indices, num_zealots, replace=False)
        self.zealots = set(zealot_indices)  # Store as a set for fast lookup

        # Initialize history arrays
        self.opinion_history = np.zeros((steps, self.N))
        self.order_history = np.zeros(steps)
        self.average_history = np.zeros(steps)
        #self.num_influencers = np.zeros(steps)
        
        # Reset opinions to the initial value
        self.opinions = self.opinions_init
        self.inf_th = inf_th
        
        # Values at t = 0
        self.opinion_history[0] = self.opinions_init
        self.average_history[0] = np.mean(self.opinions_init)
        #self.num_influencers[0] = self.count_influencers(inf_th)

        # Track number of updates
        if draw_graph:
            print("Initial Graph")
            print("Number of links: ", self.graph.number_of_edges())
            self.draw_graph(folder='Graph_Plots', filename=f'Graph_N_{self.N}_Initial')

        # Track in-degree distribution
        self.in_degree_history = {}  # Store in-degree distribution at selected time steps
        selected_steps = [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]  # Select specific timesteps
        
        self.in_degree_history[0] = dict(self.graph.in_degree())  # Store at t=0

        # Simulation loop over steps
        thres = np.random.rand(steps)
        for t in range(1, steps):
            # Update opinions with probablity 1 - phi
            #thres = np.random.rand()
            if thres[t] < 1 - phi:
                self.update_opinions()
                # Ensure zealots do not change opinion
                for z in self.zealots:
                    self.opinions[z] = 1

            # Store in-degree distribution at selected timesteps
            if t in selected_steps:
                self.in_degree_history[t] = dict(self.graph.in_degree())

            # Rewire influencers with probability phi
            if thres[t] < phi:
                num_links_before = self.graph.number_of_edges()
                self.rewire(search=search, directed=directed)
                num_links_after = self.graph.number_of_edges()

                # Check #links for consistency
                if num_links_before != num_links_after:
                    raise ValueError("The number of links changed, check the rewiring implementation!")
            
            #self.opinion_history[t] = np.copy(self.opinions)
            self.opinion_history[t] = self.opinions
            #self.average_history[t] = np.mean(self.opinions)
            #self.num_influencers[t] = self.count_influencers(inf_th)

            # Check convergence via active link search
            convergence = True
            for u, v in self.graph.edges():
                if self.opinions[u] != self.opinions[v]:
                    convergence = False

            if convergence:
                self.convergence_time = t / self.N
                # Pad remaining history arrays with the last values
                for i in range(t + 1, steps):
                    self.opinion_history[i] = self.opinion_history[t]
                    self.average_history[i] = self.average_history[t]
                    #self.num_influencers[i] = self.num_influencers[t]
                break
        
        print(f"Convergence time value: {self.convergence_time}")

        if self.convergence_time == None:
            print("Warning: Simulation did not reach equilibrium within the specified steps, set to max steps!")
            self.convergence_time = steps / self.N

        if draw_graph:
            print("Final Graph")
            self.draw_graph(folder='Graph_Plots', filename=f'Graph_N_{self.N}_Final')
        
         # Store final in-degree distribution
        self.in_degree_history[steps - 1] = dict(self.graph.in_degree())

        # Compute community sizes for the final graph
        if nx.is_directed(self.graph):
            components = nx.strongly_connected_components(self.graph)
        else:
            components = nx.connected_components(self.graph)
        
        self.component_sizes = [len(component) for component in components]

        return (self.opinion_history, self.average_history,  
                self.convergence_time, self.component_sizes)


    def simulate_two_state(self, m_int=0, steps=100, phi=None, n1_0=0.5, 
                       directed=True, search='global', 
                       force_replacement=True, draw_graph=False, f_z=0) -> dict:
        """
        Run the simulation for a two-state opinion model with rewiring updates.

        - Keeps track of the fraction of nodes in state 1.
        - Tracks the number of links between nodes in states 0-0, 0-1, and 1-1.
        - Determines convergence time based on equilibrium tolerance.

        Parameters:
            m_int: Number of influencers.
            steps: Maximum number of simulation steps.
            phi: Probability of rewiring instead of opinion update.
            directed: Whether the graph is directed.
            search: Option to let the rewiring choice be the whole graph ('global') or neigh-of-neigh ('local')
            force_replacement (float): Randomly delete link if True, otherwise only delete links of opposite opinions during
            rewiring process
            draw_graph (bool): Draw initial and final graph if True
            f_z (float): Fraction of nodes that do not change their opinion, always opinion 1

        Returns:
            results (dict): Time evolution of mean-field parameters and convergence times
        """
        # Ensure n1_0 >= f_z
        if f_z > n1_0:
            raise ValueError("f_z must be less than or equal to n1_0 to ensure enough nodes in state 1 for zealots.")

        self.m_int = m_int
        self.n1_0 = n1_0
        self.force_replacement = force_replacement
        self.convergence_time = steps / self.N

        if m_int > 0:
            self.influencers = np.array(list(range(m_int)))

        # Initialize opinions
        self.opinions = self.initialize_opinions_new(num_opinions=2, fractions=[1 - n1_0, n1_0])

        # Select zealots (f_z fraction of nodes, all having opinion 1)
        num_zealots = int(f_z * self.N)
        # Find indices of nodes already in state 1
        state_1_indices = np.where(self.opinions == 1)[0]
        # Randomly select zealots from nodes in state 1
        zealot_indices = np.random.choice(state_1_indices, num_zealots, replace=False)
        self.zealots = set(zealot_indices)  # Store as a set for fast lookup
        num_links = self.graph.number_of_edges()
        # Initialize tracking variables
        self.fraction_state_1 = np.zeros(steps)
        self.m_00 = np.zeros(steps)
        self.m_01 = np.zeros(steps)
        self.m_11 = np.zeros(steps)
        total_updates = 0
        
        if draw_graph:
            print("Initial Graph")
            self.draw_graph()
        
        thres = np.random.rand(steps)
        for t in range(steps):
            
            # Compute opinion fractions and edge statistics
            self.fraction_state_1[t] = np.mean(self.opinions)
            M_11, M_01, M_00 = 0, 0, 0
            for u, v in self.graph.edges():
                if self.opinions[u] == 0 and self.opinions[v] == 0:
                    M_00 += 1
                elif (self.opinions[u] == 0 and self.opinions[v] == 1) or (self.opinions[u] == 1 and self.opinions[v] == 0):
                    M_01 += 1
                else:
                    M_11 += 1
            
            #self.m_00[t] = M_00 / (2 * self.N)
            #self.m_01[t] = M_01 / (2 * self.N)
            #self.m_11[t] = M_11 / (2 * self.N)

            self.m_00[t] = M_00 / num_links
            self.m_01[t] = M_01 / num_links
            self.m_11[t] = M_11 / num_links
            
            # Opinion update or rewiring step
            if thres[t] < 1 - phi:  # Opinion update
                self.update_opinions()
                # Ensure zealots do not change opinion
                for z in self.zealots:
                    self.opinions[z] = 1
            else:  # Rewiring step
                self.rewire(search=search, directed=directed)
                        
            # Check convergence via active link search
            convergence = True
            for u, v in self.graph.edges():
                if self.opinions[u] != self.opinions[v]:
                    convergence = False

            if convergence:
                self.convergence_time = t / self.N
                for i in range(t + 1, steps):
                    self.fraction_state_1[i] = self.fraction_state_1[t]
                    self.m_00[i] = self.m_00[t]
                    self.m_01[i] = self.m_01[t]
                    self.m_11[i] = self.m_11[t]
                break
        
        if self.convergence_time == None:
            print("Warning: Simulation did not reach equilibrium within the specified steps, set to max steps!")
            self.convergence_time = steps / self.N

        if draw_graph:
            print("Final Graph")
            self.draw_graph()
        
         # Store final in-degree distribution
        self.in_degree_history[steps - 1] = dict(self.graph.in_degree())

        # Compute community sizes for the final graph
        if nx.is_directed(self.graph):
            components = nx.strongly_connected_components(self.graph)
        else:
            components = nx.connected_components(self.graph)
        
        self.component_sizes = [len(component) for component in components]

        return {
            'fraction_state_1': self.fraction_state_1,
            'links_00': self.m_00,
            'links_01': self.m_01,
            'links_11': self.m_11,
            'convergence_time': self.convergence_time,
        }
    
    def calculate_initial_link_counts(self, fractions=None) -> tuple[float, float, float]:
        """Calculate the initial density of links in used graph between nodes in states 0-0 and 1-1.

        Parameters:
            fractions (list): List of fractions for each opinion

        Returns:
            m11_0 (float): Initial density of links between nodes in state 1-1.
            m00_0 (float): Initial density of links between nodes in state 0-0.
            m10_0 (float): Initial density of links between nodes in state 1-0.
        """
        # Initialize opinions
        #print("n1_0 = ", self.n1_0)
        #fractions = [self.n1_0, 1 - self.n1_0]
       
        self.opinions = self.initialize_opinions_new(num_opinions=2, fractions=fractions)

        # Initialize counts
        M11_0, M00_0, M10_0 = 0, 0, 0

        # Iterate over edges in the graph
        for u, v in self.graph.edges():
            if self.opinions[u] == 1 and self.opinions[v] == 1:
                M11_0 += 1
            elif self.opinions[u] == 0 and self.opinions[v] == 0:
                M00_0 += 1
            else:
                M10_0 += 1

        m11_0 = M11_0 / (2 * self.N)
        m00_0 = M00_0 / (2 * self.N)
        m10_0 = M10_0 / (2 * self.N)
        #m10_0 = self.total_link_density - m11_0 - m00_0

        return m11_0, m00_0, m10_0
    
    # Methods for the equation-based formulation
    def odes(self, t, y) -> list[list, list, list]:
        """ODE formulation of the adaptive voter model in terms of the fraction of opinions
        in state 1 (n1(t)), the fraction of links between state 1 and 1 (m11(t)) and
        the fraction of links between state 0 and 0 (m00(t))
        """
        n1, m11, m00 = y
        m10 = self.total_link_density - m11 - m00
        n1z = self.n1z
        eps0 = 0.000006
        #p01 = m10 / (2 * m11 + m10)
        #p10 = m10 / (2 * m00 + m10)
        # Expected waiting time for a node to initiate interaction
        T = self.scale
        n1z = self.n1z
        # Full ODEs
        dn1_dt = T * (1 - self.phi) * ((1 - n1) * m10 / (2 * m00 + m10 + eps0) - (n1 - n1z) * m10 / (2 * m11 + m10 + eps0)) 
        dm11_dt = T * (self.phi * n1 * m10 / (2 * m11 + m10 + eps0) + (1 - self.phi) * (m10 * m10 / (2 * m00 + m10 + eps0) - 2 * (n1 - n1z) / n1 * m11 * m10 / (2 * m11 + m10 + eps0)) / 2)
        dm00_dt = T * (self.phi * (1 - n1) * m10 / (2 * m00 + m10 + eps0) + (1 - self.phi) * (m10 * m10 / (2 * m11 + m10 + eps0) * (n1 - n1z) / n1- 2 * m00 * m10 / (2 * m00 + m10 + eps0)) / 2)
        
        return [dn1_dt, dm11_dt, dm00_dt]

    def simulate_mean_field(self, t_span=(0, 100), phi=0, t_eval=None, n1_0=0.5, k_avg=4, m11_0=None, m00_0=None, n1z=0, scale=None) -> tuple[list, tuple]:
        """Calculate the time evolution of the equation-based formulation.

        Parameters:
            t_span (tuple): Time interval for the simulation
            phi (float): Rewiring probability 
            t_eval (list/array): Timesteps to be evaluated
            n1_0 (float): Initial fraction of opinions in state 1
            k_avg (float): Average degree of the network 
            m11_0 (float): Initial fraction of links between state 1 and 1
            m00_0 (float): Initial fraction of links between state 0 and 0
            n1z (float): Fraction of nodes that are zealots and do not change their opinion 1s
            scale (float): Scale factor in the time evolution

        Returns:
            solution (tuple): The time series (solution.t) and results of n1, m11, m10 and m00 in solution.y
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        self.phi = phi
        self.n1z = n1z
        self.total_link_density = k_avg / 2
        self.n1_0 = n1_0
        self.scale = 1 / self.N if scale == None else scale

        print(f"Using n1_0 = {n1_0}")
        # Calculate m11_0 and m00_0 from the graph if no initial values provided
        fractions = [n1_0, 1 - n1_0]
        if m11_0 is None or m00_0 is None:
            print("fractions are there boi")
            m11_0, m00_0, _ = self.calculate_initial_link_counts(fractions=fractions)
            print(f"Initial counts: m11_0: {m11_0:.2f}, m00_0: {m00_0:.2f}, m10_0: {self.total_link_density - m11_0 - m00_0:.2f}")
        #else:
        #    m11_0 = self.total_link_density / 4
        #    m00_0 = self.total_link_density / 4
        #    m10_0 = self.total_link_density / 2

        self.m00_0 = m00_0
        self.m11_0 = m11_0
        self.m10_0 = self.total_link_density - m11_0 - m00_0 
        y0 = [self.n1_0, self.m11_0, self.m00_0]
        solution = solve_ivp(self.odes, t_span, y0, t_eval=t_eval, method='LSODA')
        # Store equation-based solution
        self.eb_solution = solution
        return solution
        
    def rho_ode(self, t, rho, k_avg, phi):
        """ODE for the mean-field density of active links rho."""
        return (rho / k_avg) * ((1 - phi) * (k_avg - 1) * (1 - 2 * rho) - 1)
    
    def simulate_mean_field_rho(self, t_span=(0, 100), phi=0, t_eval=None, rho_0=0.5, k_avg=4):
        """Calculate the time evolution of the mean-field equation for \rho."""
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        self.phi = phi
        self.k_avg = k_avg

        solution = solve_ivp(
            self.rho_ode, t_span, [rho_0], args=(k_avg, phi), t_eval=t_eval, method='LSODA'
        )

        return solution
    
    def coupled_odes(self, t, y, k_avg, phi, a, b):
        """
        Defines the system of coupled ODEs for rho and c.
        
        t: time
        y: vector containing [rho, c]
        k_avg: average connectivity
        phi: external field
        a, b: parameters for the second equation
        """
        rho, c = y
        
        # Equation for d(rho)/dt
        d_rho_dt = -(2 * rho) / k_avg * (
            1 - rho + (k_avg - 1) * (rho - 1) + phi * rho + k_avg * (-1 + rho + phi - rho * phi)
        )
        
        # Equation for d(c)/dt
        d_c_dt = a * rho * (1 - c) - b * c
        
        return [d_rho_dt, d_c_dt]

    def simulate_mean_field_rho_and_c(self, t_span=(0, 100), phi=0, t_eval=None, rho_0=0.5, c_0=0.5, k_avg=4, a=1, b=1) -> tuple[list, tuple]:
        """Calculate the time evolution of the coupled mean-field equations for rho and c."""
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        initial_conditions = [rho_0, c_0]

        solution = solve_ivp(
            self.coupled_odes, t_span, initial_conditions, args=(k_avg, phi, a, b), t_eval=t_eval, method='LSODA'
        )

        return solution
    
    def draw_graph(self, folder="Graph_Plots", filename=None) -> None:
        """
        Draw the current state of the directed graph with directional arrows using the fixed layout.
        The nodes are colored based on discrete opinion values, each mapped to a unique color.
        Influencer nodes are highlighted with a thicker border.
        Eigenvector centrality values are displayed within each node.
        The top 10% of nodes by degree are highlighted with a red border.
        """
        # Define discrete opinion values and corresponding colors
        opinion_values = sorted(set(self.opinions))  # Unique opinion values
        opinion_colors = ['green', 'orange', 'blue', 'red', 'purple']  # Adjust colors as needed
        color_map = dict(zip(opinion_values, opinion_colors))

        # Assign node colors based on their opinions
        node_colors = [color_map[opinion] for opinion in self.opinions]

        # Calculate node sizes based on in-degree
        base_size = 150  # Base node size for minimum in-degree
        out_degrees = dict(self.graph.out_degree())  # Dictionary of node out-degrees
        max_out_degree = max(out_degrees.values()) if out_degrees else 1  # Avoid division by zero

        # Scale node sizes proportionally to their in-degree
        node_sizes = [base_size + (out_degrees[node] / (max_out_degree + 1)) ** 2 * 650 for node in self.graph.nodes()]

        # Calculate eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph)  # Use numpy for faster computation
        #centrality_labels = {node: f"{eigenvector_centrality[node]:.2f}" for node in self.graph.nodes()}
        centrality_labels = {node: f"{100*eigenvector_centrality[node]:.0f}" for node in self.graph.nodes()}

        # Identify top 10% nodes by degree
        num_high_degree_nodes = max(1, len(out_degrees) // 10)  # Ensure at least one node is highlighted
        top_degree_nodes = sorted(out_degrees, key=out_degrees.get, reverse=True)[:num_high_degree_nodes]

        # Prepare border width and color for nodes
        node_borders = ['red' if node in top_degree_nodes else 'black' for node in self.graph.nodes()]
        border_widths = [5 if node in top_degree_nodes else 2 for node in self.graph.nodes()]

        # Draw the graph
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        nx.draw(
            self.graph, pos=self.layout, with_labels=False, node_color=node_colors,
            arrows=True, node_size=node_sizes, edge_color='gray', font_size=12,
            edgecolors=node_borders, linewidths=border_widths
        )

        # Add labels for eigenvector centrality inside nodes
        #nx.draw_networkx_labels(
        #    self.graph, pos=self.layout, labels=centrality_labels,
        #    font_size=8, font_color="white"
        #)

        # Create a custom legend for discrete opinions
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{np.round(opinion, 0)}")
            for opinion, color in color_map.items()
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='red', markersize=10, linewidth=2)
        )

        # Ensure the plot has equal aspect ratio
        ax.set_aspect('equal', adjustable='datalim')

        # Save the plot if folder is provided
        if folder:
            # Set default filename if not provided
            if not filename:
                filename = f"Graph_plot_N{self.N}.png"

            # Ensure the directory exists
            os.makedirs(folder, exist_ok=True)

            # Full path for saving the file
            save_path = os.path.join(folder, filename)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_opinion_evolution(self, folder=None, filename=None) -> None:
        """
        Plot the opinion evolution over time, with each agent's opinion trajectory having a distinct color
        from the viridis colormap.
        """
        opinion_history = np.array(self.opinion_history)
        time_steps, num_agents = opinion_history.shape

        # Get the viridis colormap
        cmap = plt.get_cmap('viridis', num_agents)

        plt.figure(figsize=(10, 6))

        for i in range(num_agents):
            # Get the color for each agent based on its index
            agent_color = cmap(i)

            # Plot the opinion evolution with corresponding color for the agent
            plt.plot(range(time_steps), opinion_history[:, i], color=agent_color, lw=2)
 
        plt.xlabel('t')
        plt.ylabel('o(t)')
        plt.ylim(self.min_opinion - 0.1, self.max_opinion + 0.1)
        plt.grid(True)

        # Save the plot if folder is provided
        if folder:
            # Set default filename if not provided
            if not filename:
                filename = "Opinion_evolution_plot.png"
            
            # Ensure the directory exists
            os.makedirs(folder, exist_ok=True)

            # Full path for saving the file
            save_path = os.path.join(folder, filename)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def plot_average_opinion(self) -> None:
        """
        Plot the average opinion X over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.average_history, "--k", label="Average Opinion X(t)")
        plt.xlabel("t")
        plt.ylabel("X(t)")
        plt.grid(True)
        plt.ylim(self.min_opinion-0.1, self.max_opinion+0.1)
        plt.show()