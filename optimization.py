# Passenger Re-Routing Optimization During Disruptions
# Implements local search, hill climbing, and simulated annealing

import random
import math
import copy
from typing import List, Dict, Tuple, Set
from mrt_network import MRTNetwork
from search_algorithms import MRTRouteSearch


class DisruptionScenario:
    # Represents a disruption scenario with affected segments
    
    def __init__(self, name: str, disrupted_segments: List[Tuple[str, str]], 
                 reduced_frequency_segments: List[Tuple[str, str]] = None):
        self.name = name
        self.disrupted_segments = set(disrupted_segments)
        self.reduced_frequency_segments = set(reduced_frequency_segments or [])
    
    def is_segment_disrupted(self, station1: str, station2: str) -> bool:
        # Check if a segment is completely disrupted
        return (station1, station2) in self.disrupted_segments or \
               (station2, station1) in self.disrupted_segments
    
    def has_reduced_frequency(self, station1: str, station2: str) -> bool:
        # Check if a segment has reduced frequency
        return (station1, station2) in self.reduced_frequency_segments or \
               (station2, station1) in self.reduced_frequency_segments


class OptimizationProblem:
    
    # Optimization problem for passenger re-routing
    
    # State: Set of routes for multiple OD pairs
    # Objective: Minimize total travel time or maximum delay
    # Constraints: Max transfers, avoid disrupted segments, capacity limits
    
    def __init__(self, network: MRTNetwork, od_pairs: List[Tuple[str, str]], 
                 disruption: DisruptionScenario, max_transfers: int = 3):
        self.network = network
        self.od_pairs = od_pairs
        self.disruption = disruption
        self.max_transfers = max_transfers
        self.searcher = MRTRouteSearch(network)
        
        # Calculate baseline (no disruption) for comparison
        self.baseline_routes = self._calculate_baseline()
        self.baseline_cost = self._calculate_total_cost(self.baseline_routes)
    
    def _calculate_baseline(self) -> Dict[Tuple[str, str], List[str]]:
        
        # Calculate baseline routes without disruption

        routes = {}
        for origin, dest in self.od_pairs:
            result = self.searcher.a_star_search(origin, dest)
            routes[(origin, dest)] = result.path
        return routes
    
    def _calculate_total_cost(self, routes: Dict[Tuple[str, str], List[str]]) -> float:

        # Calculate total travel time across all routes

        total = 0.0
        for (origin, dest), path in routes.items():
            if path:
                cost = self._calculate_route_cost(path)
                total += cost
            else:
                total += float('inf')  # Heavily penalize infeasible routes
        return total
    
    def _calculate_route_cost(self, path: List[str]) -> float:

        # Calculate cost of a single route considering disruptions

        if not path or len(path) < 2:
            return float('inf')
        
        cost = 0.0
        for i in range(len(path) - 1):
            station1, station2 = path[i], path[i+1]
            
            # Check if segment is disrupted
            if self.disruption.is_segment_disrupted(station1, station2):
                return float('inf')  # Invalid route
            
            # Get base travel time
            neighbors = self.network.get_neighbors(station1)
            edge_cost = next((c for n, c in neighbors if n == station2), None)
            
            if edge_cost is None:
                return float('inf')  # Invalid edge
            
            cost += edge_cost
            
            # Add penalty for reduced frequency
            if self.disruption.has_reduced_frequency(station1, station2):
                cost += 2.0  # Additional 2 minutes wait time
        
        # Check transfer constraint
        transfers = self._count_transfers(path)
        if transfers > self.max_transfers:
            cost += (transfers - self.max_transfers) * 5.0  # Penalty for excess transfers
        
        return cost
    
    def _count_transfers(self, path: List[str]) -> int:
        # Count number of line transfers in a route
        if len(path) < 2:
            return 0
        
        transfers = 0
        prev_line = self.network.get_station(path[0]).line if path[0] in self.network.stations else None
        
        for station_code in path[1:]:
            station = self.network.get_station(station_code)
            if station and prev_line and station.line != prev_line:
                # Check if it's actually a different line (not just multi-line station)
                if '/' not in station.line or prev_line not in station.line:
                    transfers += 1
            if station:
                prev_line = station.line
        
        return transfers
    
    def is_route_feasible(self, path: List[str]) -> bool:
        # Check if a route is feasible under constraints
        if not path or len(path) < 2:
            return False
        
        # Check disrupted segments
        for i in range(len(path) - 1):
            if self.disruption.is_segment_disrupted(path[i], path[i+1]):
                return False
        
        # Check max transfers
        if self._count_transfers(path) > self.max_transfers:
            return False
        
        return True


class LocalSearchOptimizer:
    # Local search optimization for route planning
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
    
    def optimize(self, max_iterations: int = 100) -> Tuple[Dict[Tuple[str, str], List[str]], float, List[float]]:
        
        # Local search optimization
        
        #Neighborhood: Modify one route at a time by trying alternative paths
        
        # Start with baseline routes (modified to avoid disruptions)
        current_solution = self._get_initial_solution()
        current_cost = self.problem._calculate_total_cost(current_solution)
        
        cost_history = [current_cost]
        
        for iteration in range(max_iterations):
            # Generate neighbor by modifying one route
            neighbor = self._generate_neighbor(current_solution)
            neighbor_cost = self.problem._calculate_total_cost(neighbor)
            
            # Accept if better
            if neighbor_cost < current_cost:
                current_solution = neighbor
                current_cost = neighbor_cost
            
            cost_history.append(current_cost)
        
        return current_solution, current_cost, cost_history
    
    def _get_initial_solution(self) -> Dict[Tuple[str, str], List[str]]:
        # Get initial feasible solution
        solution = {}
        for origin, dest in self.problem.od_pairs:
            # Try to find any feasible path avoiding disruptions
            path = self._find_feasible_path(origin, dest)
            solution[(origin, dest)] = path
        return solution
    
    def _find_feasible_path(self, origin: str, dest: str) -> List[str]:
        # Find a feasible path avoiding disrupted segments
        # Use BFS to find any feasible path
        from collections import deque
        queue = deque([(origin, [origin])])
        visited = {origin}
        
        while queue:
            current, path = queue.popleft()
            
            if current == dest:
                return path
            
            for neighbor, _ in self.problem.network.get_neighbors(current):
                if neighbor not in visited:
                    if not self.problem.disruption.is_segment_disrupted(current, neighbor):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return []  # No feasible path found
    
    def _generate_neighbor(self, solution: Dict[Tuple[str, str], List[str]]) -> Dict[Tuple[str, str], List[str]]:
        # Generate a neighboring solution
        neighbor = copy.deepcopy(solution)
        
        # Pick a random OD pair to modify
        od_pair = random.choice(self.problem.od_pairs)
        origin, dest = od_pair
        
        # Try to find an alternative path
        current_path = neighbor[od_pair]
        
        if len(current_path) > 2:
            # Try removing a segment and reconnecting
            cut_point = random.randint(1, len(current_path) - 2)
            intermediate = current_path[cut_point]
            
            # Find alternative path from origin to intermediate to dest
            path1 = self._find_feasible_path(origin, intermediate)
            path2 = self._find_feasible_path(intermediate, dest)
            
            if path1 and path2:
                new_path = path1 + path2[1:]  # Avoid duplicate intermediate
                if self.problem.is_route_feasible(new_path):
                    neighbor[od_pair] = new_path
        
        return neighbor


class HillClimbingOptimizer:
    # Hill climbing with random restarts
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.local_search = LocalSearchOptimizer(problem)
    
    def optimize(self, num_restarts: int = 5, iterations_per_restart: int = 50) -> Tuple[Dict, float, List[float]]:
        # Hill climbing with random restarts
        best_solution = None
        best_cost = float('inf')
        all_costs = []
        
        for restart in range(num_restarts):
            # Random restart: perturb initial solution
            solution, cost, costs = self.local_search.optimize(max_iterations=iterations_per_restart)
            
            all_costs.extend(costs)
            
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
        
        return best_solution, best_cost, all_costs


class SimulatedAnnealingOptimizer:
    # Simulated annealing for global optimization
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.local_search = LocalSearchOptimizer(problem)
    
    def optimize(self, max_iterations: int = 200, initial_temp: float = 100.0, 
                 cooling_rate: float = 0.95) -> Tuple[Dict, float, List[float]]:
        
        # Simulated annealing optimization
        
        # Accepts worse solutions with probability based on temperature
        
        current_solution = self.local_search._get_initial_solution()
        current_cost = self.problem._calculate_total_cost(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        temperature = initial_temp
        cost_history = [current_cost]
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor = self.local_search._generate_neighbor(current_solution)
            neighbor_cost = self.problem._calculate_total_cost(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_cost - current_cost
            
            if delta < 0:
                # Always accept better solutions
                accept = True
            else:
                # Accept worse solutions with probability exp(-delta/T)
                accept_prob = math.exp(-delta / temperature)
                accept = random.random() < accept_prob
            
            if accept:
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Update best
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
            
            # Cool down
            temperature *= cooling_rate
            cost_history.append(current_cost)
        
        return best_solution, best_cost, cost_history


def run_optimization_demo():
    # Demonstrate optimization algorithms
    
    print("=== Passenger Re-Routing Optimization ===\n")
    
    # Setup: Future mode network with T5
    network = MRTNetwork(mode="future")
    
    # Disruption scenario: Tanah Merah-Expo segment suspended
    disruption = DisruptionScenario(
        "Tanah Merah-Expo Segment Suspended",
        disrupted_segments=[("EW4", "EW5")],
        reduced_frequency_segments=[("EW5", "CG2")]  # Expo to Changi Airport reduced service
    )
    
    # OD pairs focusing on Changi corridor
    od_pairs = [
        ("CG2", "EW15"),      # Changi Airport to City Hall
        ("TE32/CR1", "TE22"), # T5 to Gardens by the Bay
        ("EW10", "CG2"),      # Paya Lebar to Changi Airport
        ("TE31", "EW18"),     # Sungei Bedok to Outram Park
    ]
    
    # Create optimization problem
    problem = OptimizationProblem(network, od_pairs, disruption, max_transfers=3)
    
    print(f"Disruption: {disruption.name}")
    print(f"Number of OD pairs: {len(od_pairs)}")
    print(f"Baseline total travel time: {problem.baseline_cost:.2f} minutes\n")
    
    # Run optimization algorithms
    print("=== 1. Local Search ===")
    ls_optimizer = LocalSearchOptimizer(problem)
    ls_solution, ls_cost, ls_history = ls_optimizer.optimize(max_iterations=50)
    print(f"Final cost: {ls_cost:.2f} minutes")
    print(f"Improvement over baseline: {problem.baseline_cost - ls_cost:.2f} minutes")
    print(f"Percentage improvement: {((problem.baseline_cost - ls_cost) / problem.baseline_cost * 100):.1f}%\n")
    
    print("=== 2. Hill Climbing with Restarts ===")
    hc_optimizer = HillClimbingOptimizer(problem)
    hc_solution, hc_cost, hc_history = hc_optimizer.optimize(num_restarts=3, iterations_per_restart=30)
    print(f"Final cost: {hc_cost:.2f} minutes")
    print(f"Improvement over baseline: {problem.baseline_cost - hc_cost:.2f} minutes")
    print(f"Percentage improvement: {((problem.baseline_cost - hc_cost) / problem.baseline_cost * 100):.1f}%\n")
    
    print("=== 3. Simulated Annealing ===")
    sa_optimizer = SimulatedAnnealingOptimizer(problem)
    sa_solution, sa_cost, sa_history = sa_optimizer.optimize(max_iterations=100, initial_temp=50.0)
    print(f"Final cost: {sa_cost:.2f} minutes")
    print(f"Improvement over baseline: {problem.baseline_cost - sa_cost:.2f} minutes")
    print(f"Percentage improvement: {((problem.baseline_cost - sa_cost) / problem.baseline_cost * 100):.1f}%\n")
    
    # Show example re-routed path
    print("=== Example Re-routed Path ===")
    od = ("CG2", "EW15")
    baseline_path = problem.baseline_routes.get(od, [])
    optimized_path = sa_solution.get(od, [])
    
    print(f"OD Pair: {od[0]} → {od[1]}")
    print(f"Baseline: {' → '.join(baseline_path)}")
    print(f"Optimized: {' → '.join(optimized_path)}")
    print(f"Baseline cost: {problem._calculate_route_cost(baseline_path):.2f} min")
    print(f"Optimized cost: {problem._calculate_route_cost(optimized_path):.2f} min\n")
    
    print("=== Algorithm Comparison ===")
    print(f"{'Algorithm':<30} {'Final Cost':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Baseline (no optimization)':<30} {problem.baseline_cost:<15.2f} {'-':<15}")
    print(f"{'Local Search':<30} {ls_cost:<15.2f} {problem.baseline_cost - ls_cost:<15.2f}")
    print(f"{'Hill Climbing (restarts)':<30} {hc_cost:<15.2f} {problem.baseline_cost - hc_cost:<15.2f}")
    print(f"{'Simulated Annealing':<30} {sa_cost:<15.2f} {problem.baseline_cost - sa_cost:<15.2f}")
    
    print("\n=== Limitations ===")
    print("1. Local minima: Local search may get stuck in suboptimal solutions")
    print("2. Sensitivity to penalties: Transfer and delay penalties affect results")
    print("3. Computational cost: Larger problems require more iterations")
    print("4. Simplified model: Real-world has more complex constraints (capacity, timing)")
    print("5. Static optimization: Does not adapt to real-time changes during disruption")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    run_optimization_demo()
