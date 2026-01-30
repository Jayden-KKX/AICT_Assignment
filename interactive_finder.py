"""
MRT Route Finder
Mainly for search algos
"""

import sys
import os
from mrt_network import MRTNetwork
from search_algorithms import MRTRouteSearch, format_comparison_table


class InteractiveMRTFinder:
    """Command-line interface for MRT route finding"""
    
    def __init__(self, nodes_file: str, edges_file: str):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.network_today = None
        self.network_future = None
        self.current_network = None
        self.current_mode = "today"
        self.searcher = None
        
        self._initialize_networks()
    
    def _initialize_networks(self):
        """Load both network modes"""
        try:
            self.network_today = MRTNetwork(self.nodes_file, self.edges_file, mode="today")
            self.network_future = MRTNetwork(self.nodes_file, self.edges_file, mode="future")
            self.current_network = self.network_today
            self.searcher = MRTRouteSearch(self.current_network)
            print(f"{len(self.network_today.stations)} stations (Today Mode)")
            print(f"{len(self.network_future.stations)} stations (Future Mode)")
        except Exception as e:
            print(f"Error loading network: {e}")
            sys.exit(1)
    
    
    def print_menu(self):
        print(f"\nCurrent Mode: {self.current_mode.upper()}")
        print(f"Stations available: {len(self.current_network.stations)}\n")
        print("Options:")
        print("  1. Route between two stations")
        print("  2. Compare algorithms")
        print("  3. Algorithm workings")
        print("  4. Switch mode (Today/Future)")
        print("  5. List all stations")
        print("  6. Search for a station")
        print("  7. Exit")
        print()
    
    def switch_mode(self):
        """Switch between Today and Future modes"""
        if self.current_mode == "today":
            self.current_mode = "future"
            self.current_network = self.network_future
        else:
            self.current_mode = "today"
            self.current_network = self.network_today
        
        self.searcher = MRTRouteSearch(self.current_network)
        print(f"\nSwitched to {self.current_mode.upper()} mode")
        print(f"  Stations: {len(self.current_network.stations)}")
    
    def list_stations(self):
        """List all stations"""
        print(f"\nAll Stations in {self.current_mode.upper()} Mode:")
        print("-" * 80)
        
        stations = sorted(self.current_network.stations.values(), key=lambda s: s.name)
        for i, station in enumerate(stations, 1):
            interchange = " [INTERCHANGE]" if station.is_interchange else ""
            print(f"{i:2d}. {station.code:12s} {station.name:30s} ({station.line}){interchange}")
    
    def search_station(self):

        query = input("\nEnter station name (or part of it): ").strip()
        if not query:
            print("Invalid input.")
            return
        
        results = self.current_network.search_stations(query)
        if not results:
            print(f"No stations found matching '{query}'")
            return
        
        print(f"\nStations matching '{query}':")
        print("-" * 60)
        for code, name in results:
            station = self.current_network.get_station(code)
            interchange = " [INTERCHANGE]" if station.is_interchange else ""
            print(f"  {code:12s} {name:30s}{interchange}")
    
    def get_station_input(self, prompt: str) -> str:
        """Get station input from user"""
        while True:
            print(f"\n{prompt}")
            print("  (Enter station code (e.g. 'CG2') or station name (e.g. 'Changi Airport'))")
            user_input = input("  > ").strip()
            
            if not user_input:
                return None
            
            # Check if it's a station code
            if user_input.upper() in self.current_network.stations:
                return user_input.upper()
            
            # Check if it's a station name
            code = self.current_network.get_station_by_name(user_input)
            if code:
                return code
            
            # Search for partial matches
            results = self.current_network.search_stations(user_input)
            if len(results) == 1:
                code, name = results[0]
                confirm = input(f"  Did you mean '{name}' ({code})? (y/n): ").strip().lower()
                if confirm == 'y':
                    return code
            elif len(results) > 1:
                print(f"\n  Multiple stations found:")
                for i, (code, name) in enumerate(results, 1):
                    print(f"    {i}. {code} - {name}")
                choice = input(f"  Select station (1-{len(results)}) or 'c' to cancel: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(results):
                    return results[int(choice) - 1][0]
            
            print(f"  Station '{user_input}' not found. Try again or press Enter to cancel.")
    
    def find_route(self, algorithm: str = "A*", show_steps: bool = False):
        """Find route between two stations"""
        origin = self.get_station_input("Enter origin station:")
        if not origin:
            return
        
        destination = self.get_station_input("Enter destination station:")
        if not destination:
            return
        
        origin_name = self.current_network.get_station(origin).name
        dest_name = self.current_network.get_station(destination).name
        
        print(f"\n{'='*80}")
        print(f"Finding route: {origin_name} → {dest_name}")
        print(f"Algorithm: {algorithm}")
        print(f"{'='*80}\n")
        
        if algorithm == "A*":
            result = self.searcher.a_star_search(origin, destination, show_steps)
        elif algorithm == "BFS":
            result = self.searcher.breadth_first_search(origin, destination, show_steps)
        elif algorithm == "DFS":
            result = self.searcher.depth_first_search(origin, destination, show_steps)
        elif algorithm == "GBFS":
            result = self.searcher.greedy_best_first_search(origin, destination, show_steps)
        else:
            print(f"Unknown algorithm: {algorithm}")
            return
        
        if result.path:
            print(f"\n{'='*80}")
            print("ROUTE FOUND")
            print(f"{'='*80}")
            print(f"\nPath ({len(result.path)} stations):")
            for i, station_code in enumerate(result.path, 1):
                station = self.current_network.get_station(station_code)
                arrow = " → " if i < len(result.path) else ""
                print(f"  {i}. {station.name} ({station_code}){arrow}", end="")
            print("\n")
            print(f"Total travel time: {result.cost:.2f} minutes")
            print(f"Nodes expanded: {result.nodes_expanded}")
            print(f"Runtime: {result.runtime:.4f} seconds")
        else:
            print("\nNo route found!")
    
    def compare_algorithms(self):
        """Compare all algorithms for a route"""
        origin = self.get_station_input("Enter origin station:")
        if not origin:
            return
        
        destination = self.get_station_input("Enter destination station:")
        if not destination:
            return
        
        origin_name = self.current_network.get_station(origin).name
        dest_name = self.current_network.get_station(destination).name
        
        print(f"\n{'='*80}")
        print(f"Comparing algorithms: {origin_name} → {dest_name}")
        print(f"{'='*80}\n")
        
        results = self.searcher.compare_all_algorithms(origin, destination, show_steps=False)
        
        print("\nResults:")
        print(format_comparison_table(results))
        
        print("\n\nDetailed paths:")
        for algo_name, result in results.items():
            if result.path:
                print(f"\n{algo_name}:")
                path_names = [self.current_network.get_station(s).name for s in result.path]
                print(f"  {' → '.join(path_names[:5])}", end="")
                if len(path_names) > 5:
                    print(f" → ... → {path_names[-1]}")
                else:
                    print()
    
    def show_detailed_workings(self):
        """Show algorithm workings"""
        print("\nSelect algorithm:")
        print("  1. A* (Heuristic Search)")
        print("  2. BFS (Breadth-First)")
        print("  3. DFS (Depth-First)")
        print("  4. GBFS (Greedy Best-First)")
        
        choice = input("\nChoice (1-4): ").strip()
        algo_map = {"1": "A*", "2": "BFS", "3": "DFS", "4": "GBFS"}
        
        if choice not in algo_map:
            print("Invalid choice.")
            return
        
        self.find_route(algorithm=algo_map[choice], show_steps=True)
    
    def run(self):
        """Application loop"""
        
        while True:
            self.print_menu()
            choice = input("Select option (1-7): ").strip()
            
            if choice == "1":
                self.find_route()
            elif choice == "2":
                self.compare_algorithms()
            elif choice == "3":
                self.show_detailed_workings()
            elif choice == "4":
                self.switch_mode()
            elif choice == "5":
                self.list_stations()
            elif choice == "6":
                self.search_station()
            elif choice == "7":
                print("\nCiao!")
                break
            else:
                print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    # Get the path to the CSV files
    nodes_file = "data/nodes.csv"
    edges_file = "data/edges.csv"
    
    app = InteractiveMRTFinder(nodes_file, edges_file)
    app.run()
