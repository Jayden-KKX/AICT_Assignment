"""
Search Algorithms 
"""

from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import time
import heapq


class SearchStep:
    """Records a single step in the search process"""
    def __init__(self, step_num: int, action: str, node: str, details: Dict):
        self.step_num = step_num
        self.action = action  # 'expand', 'visit', 'goal_found', 'add_to_queue'
        self.node = node
        self.details = details  # path, cost, queue_size, heuristic, etc
    
    def __repr__(self):
        return f"Step {self.step_num}: {self.action} {self.node} - {self.details}"


class SearchResult:
    """Stores results of search algorithms"""
    def __init__(self, path: List[str], cost: float, nodes_expanded: int, 
                 runtime: float, algorithm: str, steps: List[SearchStep] = None):
        self.path = path
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.runtime = runtime
        self.algorithm = algorithm
        self.steps = steps or []
    
    def print_detailed_steps(self, network, max_steps: int = 20):
        """Print step-by-step process"""
        print(f"\n{'='*80}")
        print(f"{self.algorithm} - Step-by-Step Process")
        print(f"{'='*80}\n")
        
        shown_steps = min(len(self.steps), max_steps)
        for i, step in enumerate(self.steps[:shown_steps]):
            station_name = network.get_station(step.node).name if network.get_station(step.node) else step.node
            
            if step.action == 'expand':
                print(f"Step {step.step_num}: Expanding node '{station_name}' ({step.node})")
                print(f"  Current path: {' → '.join([network.get_station(s).name for s in step.details.get('path', [])])}")
                print(f"  Current cost: {step.details.get('cost', 0):.2f} min")
                if 'heuristic' in step.details:
                    print(f"  Heuristic to goal: {step.details['heuristic']:.2f} min")
                if 'f_value' in step.details:
                    print(f"  f(n) = g(n) + h(n) = {step.details.get('cost', 0):.2f} + {step.details.get('heuristic', 0):.2f} = {step.details['f_value']:.2f}")
                    
            elif step.action == 'add_neighbor':
                neighbor_name = network.get_station(step.details['neighbor']).name
                print(f"  → Adding neighbor: '{neighbor_name}' ({step.details['neighbor']})")
                print(f"     Edge cost: {step.details['edge_cost']:.2f} min, New total cost: {step.details['new_cost']:.2f} min")
                if 'new_heuristic' in step.details:
                    print(f"     h(neighbor): {step.details['new_heuristic']:.2f} min")
                if 'new_f' in step.details:
                    print(f"     f(neighbor): {step.details['new_f']:.2f} min")
                    
            elif step.action == 'goal_found':
                print(f"\n✓ Goal Found: '{station_name}' ({step.node})")
                print(f"  Final path: {' → '.join([network.get_station(s).name for s in step.details['path']])}")
                print(f"  Total cost: {step.details['cost']:.2f} min")
                print(f"  Nodes expanded: {step.details['nodes_expanded']}")
                break
            
            elif step.action == 'skip_visited':
                print(f"Skipping already visited: '{station_name}' ({step.node})")
            
            print()
        
        if len(self.steps) > shown_steps:
            print(f"... ({len(self.steps) - shown_steps} more steps) ...\n")
    
    def get_summary(self):
        """Summary of the search result"""
        if not self.path:
            return f"{self.algorithm}: No path found"
        
        return (f"{self.algorithm}:\n"
                f"  Path length: {len(self.path)} stations\n"
                f"  Total cost: {self.cost:.2f} minutes\n"
                f"  Nodes expanded: {self.nodes_expanded}\n"
                f"  Runtime: {self.runtime:.4f} seconds")


class MRTRouteSearch:
    """Search algorithms with logging"""
    
    def __init__(self, network, verbose: bool = False):
        self.network = network
        self.verbose = verbose
    
    def breadth_first_search(self, start: str, goal: str, show_steps: bool = False) -> SearchResult:
        """
        Breadth-First Search (BFS)
        """
        start_time = time.time()
        nodes_expanded = 0
        steps = []
        step_num = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "BFS", steps)
        
        # Queue stores (current_station, path, total_cost)
        queue = deque([(start, [start], 0.0)])
        visited: Set[str] = {start}
        
        while queue:
            current, path, cost = queue.popleft()
            step_num += 1
            nodes_expanded += 1
            
            # Record steps
            steps.append(SearchStep(step_num, 'expand', current, {
                'path': path.copy(),
                'cost': cost,
                'queue_size': len(queue)
            }))
            
            if current == goal:
                steps.append(SearchStep(step_num + 1, 'goal_found', current, {
                    'path': path,
                    'cost': cost,
                    'nodes_expanded': nodes_expanded
                }))
                runtime = time.time() - start_time
                result = SearchResult(path, cost, nodes_expanded, runtime, "BFS", steps)
                if show_steps:
                    result.print_detailed_steps(self.network)
                return result
            
            # Explore neighbors
            for neighbor, edge_cost in self.network.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    queue.append((neighbor, new_path, new_cost))
                    
                    steps.append(SearchStep(step_num, 'add_neighbor', current, {
                        'neighbor': neighbor,
                        'edge_cost': edge_cost,
                        'new_cost': new_cost
                    }))
                else:
                    steps.append(SearchStep(step_num, 'skip_visited', neighbor, {}))
        
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "BFS", steps)
    
    def depth_first_search(self, start: str, goal: str, show_steps: bool = False) -> SearchResult:
        """
        Depth-First Search (DFS)
        """
        start_time = time.time()
        nodes_expanded = 0
        steps = []
        step_num = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "DFS", steps)
        
        stack = [(start, [start], 0.0)]
        visited: Set[str] = set()
        
        while stack:
            current, path, cost = stack.pop()
            step_num += 1
            
            if current in visited:
                steps.append(SearchStep(step_num, 'skip_visited', current, {}))
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            steps.append(SearchStep(step_num, 'expand', current, {
                'path': path.copy(),
                'cost': cost,
                'stack_size': len(stack)
            }))
            
            if current == goal:
                steps.append(SearchStep(step_num + 1, 'goal_found', current, {
                    'path': path,
                    'cost': cost,
                    'nodes_expanded': nodes_expanded
                }))
                runtime = time.time() - start_time
                result = SearchResult(path, cost, nodes_expanded, runtime, "DFS", steps)
                if show_steps:
                    result.print_detailed_steps(self.network)
                return result
            
            # Explore neighbors
            neighbors = list(self.network.get_neighbors(current))
            for neighbor, edge_cost in reversed(neighbors):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    stack.append((neighbor, new_path, new_cost))
                    
                    steps.append(SearchStep(step_num, 'add_neighbor', current, {
                        'neighbor': neighbor,
                        'edge_cost': edge_cost,
                        'new_cost': new_cost
                    }))
        
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "DFS", steps)
    
    def greedy_best_first_search(self, start: str, goal: str, show_steps: bool = False) -> SearchResult:
        """
        Greedy Best-First Search (GBFS)
        """
        start_time = time.time()
        nodes_expanded = 0
        steps = []
        step_num = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "GBFS", steps)
        
        h_start = self.network.heuristic(start, goal)
        pq = [(h_start, start, [start], 0.0)]
        visited: Set[str] = set()
        
        while pq:
            h, current, path, cost = heapq.heappop(pq)
            step_num += 1
            
            if current in visited:
                steps.append(SearchStep(step_num, 'skip_visited', current, {}))
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            steps.append(SearchStep(step_num, 'expand', current, {
                'path': path.copy(),
                'cost': cost,
                'heuristic': h,
                'pq_size': len(pq)
            }))
            
            if current == goal:
                steps.append(SearchStep(step_num + 1, 'goal_found', current, {
                    'path': path,
                    'cost': cost,
                    'nodes_expanded': nodes_expanded
                }))
                runtime = time.time() - start_time
                result = SearchResult(path, cost, nodes_expanded, runtime, "GBFS", steps)
                if show_steps:
                    result.print_detailed_steps(self.network)
                return result
            
            for neighbor, edge_cost in self.network.get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    new_h = self.network.heuristic(neighbor, goal)
                    heapq.heappush(pq, (new_h, neighbor, new_path, new_cost))
                    
                    steps.append(SearchStep(step_num, 'add_neighbor', current, {
                        'neighbor': neighbor,
                        'edge_cost': edge_cost,
                        'new_cost': new_cost,
                        'new_heuristic': new_h
                    }))
        
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "GBFS", steps)
    
    def a_star_search(self, start: str, goal: str, show_steps: bool = False) -> SearchResult:
        """
        A* Search
        """
        start_time = time.time()
        nodes_expanded = 0
        steps = []
        step_num = 0
        
        if start not in self.network.stations or goal not in self.network.stations:
            return SearchResult([], float('inf'), 0, 0.0, "A*", steps)
        
        h_start = self.network.heuristic(start, goal)
        pq = [(h_start, 0.0, start, [start])]
        g_values: Dict[str, float] = {start: 0.0}
        
        while pq:
            f, g, current, path = heapq.heappop(pq)
            step_num += 1
            nodes_expanded += 1
            
            h = f - g
            steps.append(SearchStep(step_num, 'expand', current, {
                'path': path.copy(),
                'cost': g,
                'heuristic': h,
                'f_value': f,
                'pq_size': len(pq)
            }))
            
            if g > g_values.get(current, float('inf')):
                continue
            
            if current == goal:
                steps.append(SearchStep(step_num + 1, 'goal_found', current, {
                    'path': path,
                    'cost': g,
                    'nodes_expanded': nodes_expanded
                }))
                runtime = time.time() - start_time
                result = SearchResult(path, g, nodes_expanded, runtime, "A*", steps)
                if show_steps:
                    result.print_detailed_steps(self.network)
                return result
            
            for neighbor, edge_cost in self.network.get_neighbors(current):
                new_g = g + edge_cost
                
                if new_g < g_values.get(neighbor, float('inf')):
                    g_values[neighbor] = new_g
                    new_path = path + [neighbor]
                    new_h = self.network.heuristic(neighbor, goal)
                    new_f = new_g + new_h
                    heapq.heappush(pq, (new_f, new_g, neighbor, new_path))
                    
                    steps.append(SearchStep(step_num, 'add_neighbor', current, {
                        'neighbor': neighbor,
                        'edge_cost': edge_cost,
                        'new_cost': new_g,
                        'new_heuristic': new_h,
                        'new_f': new_f
                    }))
        
        runtime = time.time() - start_time
        return SearchResult([], float('inf'), nodes_expanded, runtime, "A*", steps)
    
    def compare_all_algorithms(self, start: str, goal: str, show_steps: bool = False) -> Dict[str, SearchResult]:
        """Run algorithms comparison"""
        results = {
            "BFS": self.breadth_first_search(start, goal, show_steps),
            "DFS": self.depth_first_search(start, goal, show_steps),
            "GBFS": self.greedy_best_first_search(start, goal, show_steps),
            "A*": self.a_star_search(start, goal, show_steps)
        }
        return results


def format_comparison_table(results: Dict[str, SearchResult]) -> str:
    """Comparison table"""
    header = f"{'Algorithm':<10} {'Cost (min)':<12} {'Nodes':<10} {'Runtime (s)':<12} {'Path Length':<12}"
    separator = "-" * 66
    
    lines = [header, separator]
    for algo_name, result in results.items():
        path_len = len(result.path) if result.path else 0
        cost_str = f"{result.cost:.2f}" if result.cost != float('inf') else "No path"
        line = f"{algo_name:<10} {cost_str:<12} {result.nodes_expanded:<10} {result.runtime:<12.4f} {path_len:<12}"
        lines.append(line)
    
    return "\n".join(lines)
