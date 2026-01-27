import os
import pandas as pd
from collections import deque

# --- DIRECTORY SETUP ---
# Gets the directory where this script (bfs_search.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up one level to find edges.csv in the parent folder
parent_dir = os.path.dirname(script_dir)
edges_path = os.path.join(parent_dir, 'edges.csv')

def load_mrt_graph(file_path, mode="Today"):
    """
    Loads MRT network data into a directed graph.
    As per assignment: 
    - Today Mode uses current connections[cite: 32, 57].
    - Future Mode includes TELe and CRL extensions [cite: 33, 58-61].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical Error: Could not find {file_path}")

    df = pd.read_csv(file_path)
    
    # Filter for Today Mode: Exclude extensions (Edge_ID >= 34)
    if mode == "Today":
        df = df[df['Edge_ID'] < 34]
    
    # Build Directed Adjacency List
    graph = {}
    for _, row in df.iterrows():
        u = str(row['From_Station']).strip()
        v = str(row['To_Station']).strip()
        
        # Consistent cost function: time + penalties [cite: 52]
        weight = float(row['Total_Edge_Weight_Min'])
        
        if u not in graph:
            graph[u] = []
        
        # Directed edge only (Assume no bidirectional travel)
        graph[u].append((v, weight))
        
    return graph

def bfs_search(graph, start, goal):
    """
    Implements Breadth-First Search to find the shortest path by hops[cite: 48].
    Tracks expanded nodes and total path cost for analysis[cite: 54].
    """
    # Queue stores: (current_node, path_list, cumulative_cost)
    queue = deque([(start, [start], 0)])
    visited = {start}
    nodes_expanded = 0
    
    while queue:
        current_node, path, current_cost = queue.popleft()
        nodes_expanded += 1
        
        # Goal Check
        if current_node == goal:
            return path, current_cost, nodes_expanded
        
        # Explore neighbors
        for neighbor, weight in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], current_cost + weight))
                
    return None, 0, nodes_expanded

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Change this to "Future" to enable Changi Terminal 5 connections [cite: 58-61]
    current_mode = "Future" 
    
    try:
        mrt_graph = load_mrt_graph(edges_path, mode=current_mode)
        
        # Recommended test pair: Paya Lebar to Changi Terminal 5 [cite: 70]
        start = "Changi Airport"
        goal = "MacPherson"
        
        print(f"--- ChangiLink AI: BFS Routing ({current_mode} Mode) ---")
        path, cost, expanded = bfs_search(mrt_graph, start, goal)
        
        if path:
            print(f"Path Result: {' -> '.join(path)}")
            print(f"Total Travel Time: {cost:.2f} mins")
            print(f"Performance: {expanded} nodes expanded")
        else:
            print(f"No path found from {start} to {goal}.")
            if current_mode == "Today":
                print("Note: Changi Terminal 5 is only available in Future Mode.")
                
    except Exception as e:
        print(f"Error: {e}")