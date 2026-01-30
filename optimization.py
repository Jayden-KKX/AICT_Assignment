"""
Passenger Re-Routing Optimization During Disruptions
Implements local search, hill climbing, and simulated annealing
Disruption scenario testing
"""

import random
import math
import copy
import sys

from typing import List, Dict, Tuple, Set
from mrt_network import MRTNetwork
from search_algorithms import MRTRouteSearch


class OptimizationStep:
    """Records a step in the optimization process"""
    def __init__(self, iteration: int, current_cost: float, best_cost: float, 
                 temperature: float = None, accepted: bool = None, delta: float = None):
        self.iteration = iteration
        self.current_cost = current_cost
        self.best_cost = best_cost
        self.temperature = temperature
        self.accepted = accepted
        self.delta = delta


class DisruptionScenario:
    """Represents a disruption scenario with affected segments"""
    
    def __init__(self, name: str, disrupted_segments: List[Tuple[str, str]], 
                 reduced_frequency_segments: List[Tuple[str, str]] = None):
        self.name = name
        self.disrupted_segments = set(disrupted_segments)
        self.reduced_frequency_segments = set(reduced_frequency_segments or [])
    
    def is_segment_disrupted(self, station1: str, station2: str) -> bool:
        """Check if a segment is completely disrupted"""
        return (station1, station2) in self.disrupted_segments or \
               (station2, station1) in self.disrupted_segments
    
    def has_reduced_frequency(self, station1: str, station2: str) -> bool:
        """Check if a segment has reduced frequency"""
        return (station1, station2) in self.reduced_frequency_segments or \
               (station2, station1) in self.reduced_frequency_segments
    
    def print_summary(self):
        """Print disruption summary"""
        print(f"\nDisruption: {self.name}")
        print(f"  Disrupted segments: {len(self.disrupted_segments)}")
        for s1, s2 in self.disrupted_segments:
            print(f"    • {s1} ↔ {s2}")
        if self.reduced_frequency_segments:
            print(f"  Reduced frequency: {len(self.reduced_frequency_segments)}")
            for s1, s2 in self.reduced_frequency_segments:
                print(f"    • {s1} ↔ {s2}")


class OptimizationProblem:
    """
    Optimization problem for passenger re-routing
    
    State: Set of routes for multiple OD pairs
    Objective: Minimize total travel time
    Constraints: Max transfers, avoid disrupted segments
    """
    
    def __init__(self, network: MRTNetwork, od_pairs: List[Tuple[str, str]], 
                 disruption: DisruptionScenario, max_transfers: int = 3):
        self.network = network
        self.od_pairs = od_pairs
        self.disruption = disruption
        self.max_transfers = max_transfers
        self.searcher = MRTRouteSearch(network)
        
        # Calculate baseline (no disruption)
        self.baseline_routes = self._calculate_baseline()
        self.baseline_cost = self._calculate_total_cost(self.baseline_routes)
    
    def _calculate_baseline(self) -> Dict[Tuple[str, str], List[str]]:
        """Calculate baseline routes without disruption"""
        routes = {}
        for origin, dest in self.od_pairs:
            result = self.searcher.a_star_search(origin, dest)
            routes[(origin, dest)] = result.path
        return routes
    
    def _calculate_total_cost(self, routes: Dict[Tuple[str, str], List[str]]) -> float:
        """Calculate total travel time across all routes"""
        total = 0.0
        for (origin, dest), path in routes.items():
            if path:
                cost = self._calculate_route_cost(path)
                total += cost
            else:
                total += float('inf')
        return total
    
    def _calculate_route_cost(self, path: List[str]) -> float:
        """Calculate cost of a single route considering disruptions"""
        if not path or len(path) < 2:
            return float('inf')
        
        cost = 0.0
        for i in range(len(path) - 1):
            station1, station2 = path[i], path[i+1]
            
            # Check if segment is disrupted
            if self.disruption.is_segment_disrupted(station1, station2):
                return float('inf')
            
            # Get base travel time
            neighbors = self.network.get_neighbors(station1)
            edge_cost = next((c for n, c in neighbors if n == station2), None)
            
            if edge_cost is None:
                return float('inf')
            
            cost += edge_cost
            
            # Add penalty for reduced frequency
            if self.disruption.has_reduced_frequency(station1, station2):
                cost += 2.0
        
        # Check transfer constraint
        transfers = self._count_transfers(path)
        if transfers > self.max_transfers:
            cost += (transfers - self.max_transfers) * 5.0
        
        return cost
    
    def _count_transfers(self, path: List[str]) -> int:
        """Count number of line transfers in a route"""
        if len(path) < 2:
            return 0
        
        transfers = 0
        prev_line = self.network.get_station(path[0]).line if path[0] in self.network.stations else None
        
        for station_code in path[1:]:
            station = self.network.get_station(station_code)
            if station and prev_line and station.line != prev_line:
                if '/' not in station.line or prev_line not in station.line:
                    transfers += 1
            if station:
                prev_line = station.line
        
        return transfers
    
    def is_route_feasible(self, path: List[str]) -> bool:
        """Check if a route is feasible under constraints"""
        if not path or len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            if self.disruption.is_segment_disrupted(path[i], path[i+1]):
                return False
        
        if self._count_transfers(path) > self.max_transfers:
            return False
        
        return True
    
    def print_solution_details(self, routes: Dict[Tuple[str, str], List[str]], 
                                title: str = "Solution"):
        """Print detailed solution information"""
        
        total_cost = 0.0
        for i, ((origin, dest), path) in enumerate(routes.items(), 1):
            cost = self._calculate_route_cost(path)
            total_cost += cost
            
            origin_name = self.network.get_station(origin).name if origin in self.network.stations else origin
            dest_name = self.network.get_station(dest).name if dest in self.network.stations else dest
            
            print(f"Route {i}: {origin_name} → {dest_name}")
            if path and cost < float('inf'):
                path_names = [self.network.get_station(s).name for s in path if s in self.network.stations]
                print(f"  Path: {' → '.join(path_names[:5])}", end="")
                if len(path_names) > 5:
                    print(f" → ... → {path_names[-1]}")
                else:
                    print()
                print(f"  Cost: {cost:.2f} min, Stations: {len(path)}, Transfers: {self._count_transfers(path)}")
            else:
                print(f"  No feasible route")
            print()
        
        print(f"Total cost: {total_cost:.2f} min")


class SimulatedAnnealingOptimizer:
    """ Simulated annealing with steps"""
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
    
    def optimize(self, max_iterations: int = 200, initial_temp: float = 100.0, 
                 cooling_rate: float = 0.95, show_steps: bool = False) -> Tuple[Dict, float, List[OptimizationStep]]:
        """
        Simulated annealing optimization with logging
        
        Returns:
            (best_solution, best_cost, optimization_steps)
        """
        # Get initial solution
        current_solution = self._get_initial_solution()
        current_cost = self.problem._calculate_total_cost(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        temperature = initial_temp
        steps = []
        
        if show_steps:
            print("SIMULATED ANNEALING")
            print(f"Parameters:")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Initial temperature: {initial_temp}")
            print(f"  Cooling rate: {cooling_rate}")
            print(f"\nInitial solution cost: {current_cost:.2f} min")
            print(f"Baseline cost: {self.problem.baseline_cost:.2f} min\n")
            
            print(f"{'Iter':<6} {'Current':<10} {'Best':<10} {'Temp':<10} {'Delta':<10} {'Accept?':<8}")
            print("-" * 70)
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor = self._generate_neighbor(current_solution)
            neighbor_cost = self.problem._calculate_total_cost(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_cost - current_cost
            
            if delta < 0:
                # Always accept better solutions
                accept = True
                accept_prob = 1.0
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
            
            # Record step
            step = OptimizationStep(iteration, current_cost, best_cost, temperature, accept, delta)
            steps.append(step)
            
            # Show progress
            if show_steps and (iteration < 10 or iteration % 20 == 0):
                accept_str = "ACCEPTED" if accept else "REJECTED"
                print(f"{iteration:<6} {current_cost:<10.2f} {best_cost:<10.2f} {temperature:<10.2f} {delta:<10.2f} {accept_str:<8}")
            
            # Cool down
            temperature *= cooling_rate
        
        if show_steps:
            print("-" * 70)
            print(f"Final best cost: {best_cost:.2f} min")
            print(f"Baseline cost: {self.problem.baseline_cost:.2f} min")
            print(f"Improvement: {self.problem.baseline_cost - best_cost:.2f} min ({((self.problem.baseline_cost - best_cost) / self.problem.baseline_cost * 100):.1f}%)")
        
        return best_solution, best_cost, steps
    
    def _get_initial_solution(self) -> Dict[Tuple[str, str], List[str]]:
        """Get initial feasible solution"""
        solution = {}
        for origin, dest in self.problem.od_pairs:
            path = self._find_feasible_path(origin, dest)
            solution[(origin, dest)] = path
        return solution
    
    def _find_feasible_path(self, origin: str, dest: str) -> List[str]:
        """Find a feasible path avoiding disrupted segments"""
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
        
        return []
    
    def _generate_neighbor(self, solution: Dict[Tuple[str, str], List[str]]) -> Dict[Tuple[str, str], List[str]]:
        """Generate a neighboring solution"""
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
            
            # Find alternative path
            path1 = self._find_feasible_path(origin, intermediate)
            path2 = self._find_feasible_path(intermediate, dest)
            
            if path1 and path2:
                new_path = path1 + path2[1:]
                if self.problem.is_route_feasible(new_path):
                    neighbor[od_pair] = new_path
        
        return neighbor


class InteractiveOptimization:
    """Interactive interface for optimization"""
    
    def __init__(self, nodes_file: str, edges_file: str):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.network = None
        self.mode = "future"
    
    def run(self):
        """Run interactive mode"""
        print("INTERACTIVE OPTIMIZATION SYSTEM")
        
        # Load network
        print("\nLoading network...")
        self.network = MRTNetwork(self.nodes_file, self.edges_file, mode=self.mode)
        print(f"Loaded {len(self.network.stations)} stations ({self.mode.upper()} mode)\n")
        
        while True:
            print("\n\nOptions:")
            print("  1. Test predefined disruption scenario")
            print("  2. Create custom disruption scenario")
            print("  3. Run simulated annealing with visualization")
            print("  4. Compare baseline vs optimized routes")
            print("  5. Switch mode (Today/Future)")
            print("  6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.predefined_scenario()
            elif choice == '2':
                self.custom_scenario()
            elif choice == '3':
                self.run_sa_detailed()
            elif choice == '4':
                self.compare_routes()
            elif choice == '5':
                self.switch_mode()
            elif choice == '6':
                print("\nThank you for using the Optimization System!")
                break
            else:
                print("\nInvalid option. Please try again.")
    
    def predefined_scenario(self):
        """Test predefined disruption scenarios"""
        
        scenarios = [
            DisruptionScenario(
                "Tanah Merah-Expo Suspended",
                [("EW4", "CG1/DT35")],
                [("CG1/DT35", "CG2")]
            ),
            DisruptionScenario(
                "Major Interchange Closure",
                [("EW16/NE3/DT19/TE17", "EW15")],
                []
            ),
        ]
        
        print("Available scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario.name}")
        
        choice = input("\nSelect scenario (1-2): ").strip()
        
        if choice not in ['1', '2']:
            print("\nInvalid choice.")
            return
        
        scenario = scenarios[int(choice) - 1]
        scenario.print_summary()
        
        # Define OD pairs
        od_pairs = [
            ("CG2", "EW12/DT14"),
            ("EW8/CC9", "CG2"),
        ]
        
        if self.mode == "future" and "TE32/CR1" in self.network.stations:
            od_pairs.append(("TE32/CR1", "TE22/DT18"))
        
        print(f"\nOD pairs: {len(od_pairs)}")
        for origin, dest in od_pairs:
            o_name = self.network.get_station(origin).name if origin in self.network.stations else origin
            d_name = self.network.get_station(dest).name if dest in self.network.stations else dest
            print(f"  • {o_name} → {d_name}")
        
        # Create problem
        problem = OptimizationProblem(self.network, od_pairs, scenario, max_transfers=3)
        
        print(f"\nBaseline total cost: {problem.baseline_cost:.2f} min")
        
        # Optimize
        print("\nRunning optimization...")
        optimizer = SimulatedAnnealingOptimizer(problem)
        solution, cost, steps = optimizer.optimize(max_iterations=100, show_steps=True)
        
        # Show solution
        problem.print_solution_details(solution, "Optimized Routes")
        
        input("\nPress Enter to continue...")
    
    def custom_scenario(self):
        """Create disruption scenario"""
        
        input("\nPress Enter to continue...")
    
    def run_sa_detailed(self):
        """Run SA with detailed visualization"""
        
        # Use simple scenario
        scenario = DisruptionScenario(
            "Test Scenario",
            [("EW4", "CG1/DT35")],
            []
        )
        
        od_pairs = [
            ("CG2", "EW12/DT14"),
            ("EW8/CC9", "CG2"),
        ]
        
        problem = OptimizationProblem(self.network, od_pairs, scenario)
        
        print(f"Disruption: {scenario.name}")
        print(f"OD pairs: {len(od_pairs)}")
        print(f"Baseline: {problem.baseline_cost:.2f} min\n")
        
        # Get parameters
        try:
            iters = int(input("Max iterations (default 100): ").strip() or "100")
            temp = float(input("Initial temperature (default 100.0): ").strip() or "100.0")
            cool = float(input("Cooling rate (default 0.95): ").strip() or "0.95")
        except:
            print("\nUsing default parameters.")
            iters, temp, cool = 100, 100.0, 0.95
        
        # Run
        optimizer = SimulatedAnnealingOptimizer(problem)
        solution, cost, steps = optimizer.optimize(iters, temp, cool, show_steps=True)
        
        input("\nPress Enter to continue...")
    
    def compare_routes(self):
        """Compare baseline vs optimized routes"""
        
        scenario = DisruptionScenario(
            "Test Scenario",
            [("EW4", "CG1/DT35")],
            []
        )
        
        od_pairs = [
            ("CG2", "EW12/DT14"),
        ]
        
        problem = OptimizationProblem(self.network, od_pairs, scenario)
        
        print("BASELINE ROUTES (No disruption handling):")
        problem.print_solution_details(problem.baseline_routes, "Baseline")
        
        optimizer = SimulatedAnnealingOptimizer(problem)
        solution, cost, _ = optimizer.optimize(max_iterations=100, show_steps=False)
        
        problem.print_solution_details(solution, "Optimized Routes")
        
        input("\nPress Enter to continue...")
    
    def switch_mode(self):
        """Switch between modes"""
        self.mode = "today" if self.mode == "future" else "future"
        self.network = MRTNetwork(self.nodes_file, self.edges_file, mode=self.mode)
        print(f"\n✓ Switched to {self.mode.upper()} mode")
        print(f"  Stations: {len(self.network.stations)}")


if __name__ == "__main__":
    app = InteractiveOptimization(
        "data/nodes.csv",
        "data/edges.csv"
    )
    app.run()
