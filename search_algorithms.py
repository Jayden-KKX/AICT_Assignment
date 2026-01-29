"""
Search Algorithms for MRT Route Planning
BFS, DFS, Greedy Best-First Search, A* Search
"""

from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import time
from mrt_network import MRTNetwork
import heapq


class SearchResult:
    """Stores results of search algorithms"""
    def __init__(self, path: List[str], cost: float, nodes_expanded: int, 
                 runtime: float, algorithm: str):
        self.path = path
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.runtime = runtime
        self.algorithm = algorithm
    
    def __repr__(self):
        path_str = " -> ".join(self.path) if self.path else "No path found"
        return (f"{self.algorithm}:\n"
                f"  Path: {path_str}\n"
                f"  Cost: {self.cost:.2f} minutes\n"
                f"  Nodes expanded: {self.nodes_expanded}\n"
                f"  Runtime: {self.runtime:.4f} seconds\n")


class MRTRouteSearch:
    # Search algorithms for finding routes in MRT network
    
    def __init__(self, network: MRTNetwork):
        self.network = network
    
    def breadth_first_search(self, start: str, goal: str) -> SearchResult:
        
        # Breadth-First Search (BFS)
        
        # Explores nodes level by level using a queue (FIFO method)
        # Guarantees shortest path in terms of number of stations
        # Is not necessarily optimal in terms of travel time

        start_time = time.time()
        nodes_expanded = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "BFS")
        
        # Queue stores (current_station, path, total_cost)
        queue = deque([(start, [start], 0.0)])
        visited: Set[str] = {start}
        
        while queue:
            current, path, cost = queue.popleft()
            nodes_expanded += 1
            
            if current == goal:
                runtime = time.time() - start_time
                return SearchResult(path, cost, nodes_expanded, runtime, "BFS")
            
            # Explore neighbors
            for neighbor, edge_cost in self.network.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    queue.append((neighbor, new_path, new_cost))
        
        # No path found
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "BFS")
    
    def depth_first_search(self, start: str, goal: str) -> SearchResult:

        # Depth-First Search (DFS)
        
        # Explores as far as possible along each branch using a stack (LIFO).
        # Not guaranteed to find optimal path. May take longer routes.

        start_time = time.time()
        nodes_expanded = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "DFS")
        
        # Stack stores (current_station, path, total_cost)
        stack = [(start, [start], 0.0)]
        visited: Set[str] = set()
        
        while stack:
            current, path, cost = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            if current == goal:
                runtime = time.time() - start_time
                return SearchResult(path, cost, nodes_expanded, runtime, "DFS")
            
            # Explore neighbors (reversed to maintain reasonable order)
            neighbors = list(self.network.get_neighbors(current))
            for neighbor, edge_cost in reversed(neighbors):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    stack.append((neighbor, new_path, new_cost))
        
        # No path found
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "DFS")
    
    def greedy_best_first_search(self, start: str, goal: str) -> SearchResult:
        
        # Greedy Best-First Search (GBFS)
        
        # Uses heuristic (straight-line distance) to guide search toward goal.
        # Expands node that appears closest to goal first.
        # Not guaranteed to find optimal path.
        
        # Heuristic: Straight-line distance converted to approximate travel time

        start_time = time.time()
        nodes_expanded = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "GBFS")
        
        # Priority queue stores (heuristic_value, current_station, path, total_cost)

        pq = [(self.network.heuristic(start, goal), start, [start], 0.0)]
        visited: Set[str] = set()
        
        while pq:
            _, current, path, cost = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            if current == goal:
                runtime = time.time() - start_time
                return SearchResult(path, cost, nodes_expanded, runtime, "GBFS")
            
            # Explore neighbors
            for neighbor, edge_cost in self.network.get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    h = self.network.heuristic(neighbor, goal)
                    heapq.heappush(pq, (h, neighbor, new_path, new_cost))
        
        # No path found
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "GBFS")
    
    def a_star_search(self, start: str, goal: str) -> SearchResult:
        ""
        # A* Search
        
        # Combines actual cost (g) and heuristic (h) to find optimal path.
        # Expands node with lowest f = g + h value.
        # Guaranteed to find optimal path if heuristic is admissible.
        
        # g(n): actual travel time from start to node n
        # h(n): estimated travel time from node n to goal (straight-line distance)
        # f(n) = g(n) + h(n): estimated total cost

        start_time = time.time()
        nodes_expanded = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "A*")
        
        # Priority queue stores (f_value, g_value, current_station, path)
        h_start = self.network.heuristic(start, goal)
        pq = [(h_start, 0.0, start, [start])]
        
        # Track best g-value seen for each node
        g_values: Dict[str, float] = {start: 0.0}
        
        while pq:
            f, g, current, path = heapq.heappop(pq)
            nodes_expanded += 1
            
            # Skip if we've found a better path to this node
            if g > g_values.get(current, float('inf')):
                continue
            
            if current == goal:
                runtime = time.time() - start_time
                return SearchResult(path, g, nodes_expanded, runtime, "A*")
            
            # Explore neighbors
            for neighbor, edge_cost in self.network.get_neighbors(current):
                new_g = g + edge_cost
                
                # Only proceed if this is a better path
                if new_g < g_values.get(neighbor, float('inf')):
                    g_values[neighbor] = new_g
                    new_path = path + [neighbor]
                    h = self.network.heuristic(neighbor, goal)
                    new_f = new_g + h
                    heapq.heappush(pq, (new_f, new_g, neighbor, new_path))
        
        # No path found
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "A*")
    
    def compare_all_algorithms(self, start: str, goal: str) -> Dict[str, SearchResult]:
        """Run all search algorithms and return comparison"""
        results = {
            "BFS": self.breadth_first_search(start, goal),
            "DFS": self.depth_first_search(start, goal),
            "GBFS": self.greedy_best_first_search(start, goal),
            "A*": self.a_star_search(start, goal)
        }
        return results


def format_comparison_table(results: Dict[str, SearchResult]) -> str:
    
    # Format results as comparison table

    header = f"{'Algorithm':<10} {'Cost (min)':<12} {'Nodes':<10} {'Runtime (s)':<12} {'Path Length':<12}"
    separator = "-" * 66
    
    lines = [header, separator]
    for algo_name, result in results.items():
        path_len = len(result.path) if result.path else 0
        cost_str = f"{result.cost:.2f}" if result.cost != float('inf') else "No path"
        line = f"{algo_name:<10} {cost_str:<12} {result.nodes_expanded:<10} {result.runtime:<12.4f} {path_len:<12}"
        lines.append(line)
    
    return "\n".join(lines)


# Test the search algorithms
if __name__ == "__main__":
    print("=== Testing Search Algorithms ===\n")
    
    # Test with Today Mode
    network_today = MRTNetwork(mode="today")
    searcher = MRTRouteSearch(network_today)
    
    print("Test Case: Changi Airport (CG2) to City Hall (EW15) - Today Mode\n")
    results = searcher.compare_all_algorithms("CG2", "EW15")
    
    for algo_name, result in results.items():
        print(result)
    
    print("\n" + format_comparison_table(results))
    
    # Test with Future Mode
    print("\n\n=== Future Mode with Terminal 5 ===\n")
    network_future = MRTNetwork(mode="future")
    searcher_future = MRTRouteSearch(network_future)
    
    print("Test Case: Changi Terminal 5 (TE32/CR1) to Gardens by the Bay (TE22) - Future Mode\n")
    results_future = searcher_future.compare_all_algorithms("TE32/CR1", "TE22")
    
    for algo_name, result in results_future.items():
        print(result)
    
    print("\n" + format_comparison_table(results_future))

