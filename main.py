import sys

from mrt_network import MRTNetwork
from search_algorithms import MRTRouteSearch, format_comparison_table
from logical_inference import InteractiveLogicalInference
from optimization import InteractiveOptimization


def print_header(title: str):
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + title.center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")


def print_section(title: str):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def requirement_1_search_algorithms():
    """Q1: Search Algorithms"""
    print_header("Q1: SEARCH ALGORITHMS")
    
    print("\n\nMenu:")
    print("  1. Quick example")
    print("  2. Route finder")
    print("  3. Compare algorithms")
    print("  4. Main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Quick example
        print_section("Quick Example: A* with Steps")
        
        network = MRTNetwork(
            "data/nodes.csv",
            "data/edges.csv",
            mode="today"
        )
        
        searcher = MRTRouteSearch(network)
        
        print("Finding route: Changi Airport → City Hall")
        print("Using A* algorithm\n")
        
        result = searcher.a_star_search("CG2", "EW12/DT14", show_steps=True)
        
        input("\nPress Enter to continue...")
        
    elif choice == '2':
        # Interactive finder
        from interactive_finder import InteractiveMRTFinder
        
        app = InteractiveMRTFinder(
            "data/nodes.csv",
            "data/edges.csv"
        )
        app.run()
        
    elif choice == '3':
        # Compare algorithms
        print_section("Algorithm Comparison")
        
        network = MRTNetwork(
            "data/nodes.csv",
            "data/edges.csv",
            mode="today"
        )
        
        searcher = MRTRouteSearch(network)
        
        print("Route: Changi Airport → City Hall\n")
        
        results = searcher.compare_all_algorithms("CG2", "EW12/DT14")
        
        print(format_comparison_table(results))
        
        print("\n\nPaths:")
        for algo_name, result in results.items():
            if result.path:
                path_names = [network.get_station(s).name for s in result.path]
                print(f"\n{algo_name}:")
                print(f"  {' → '.join(path_names)}")
        
        input("\n\nPress Enter to continue...")


def requirement_2_logical_inference():
    """Q2: Logical Inference"""
    print_header("Q2: LOGICAL INFERENCE")
    
    print("\n\nMenu:")
    print("  1. Quick example")
    print("  2. Interactive logical inference")
    print("  3. Back to main menu")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        # Quick example
        print_section("Consistency Check")
        
        from logical_inference import initialize_mrt_rules, Clause
        
        engine = initialize_mrt_rules(mode="future")
        
        advisory = {
            Clause({"systems_integration"}),
            Clause({"¬service_EW4_EW5"})
        }
        
        consistent, msg, steps = engine.is_consistent(advisory, show_steps=True)
        
        input("\nPress Enter to continue...")
        
    elif choice == '2':
        # Interactive
        app = InteractiveLogicalInference()
        app.run()


def requirement_3_bayesian_network():
    """Q3: Bayesian Network"""
    print_header("Q3: BAYESIAN NETWORK")
    
    print("\n\nMenu:")
    print("  1. Run all required scenarios")
    print("  2. Quick single example")
    print("  3. Back to main menu")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        print("\nRunning all required scenarios...\n")
        
        try:
            from bayesian_network import run_required_scenarios
            run_required_scenarios()
        except ImportError:
            from bayesian_network import run_crowding_risk_scenarios
            run_crowding_risk_scenarios()
        
        input("\nPress Enter to continue...")
        
    elif choice == '2':
        # Quick example
        print_section("Quick Example: Single Scenario")
        
        try:
            # Try improved version
            from bayesian_network import BayesianNetwork
            
            bn = BayesianNetwork()
            
            # Run a single scenario with the improved format
            bn.run_scenario(
                "Clear Evening Weekday + Normal Service (FUTURE)",
                {
                    'Weather': 'clear',
                    'Time_Of_Day': 'evening',
                    'Day_Type': 'weekday',
                    'Network_Mode': 'future',
                    'Service_Status': 'normal'
                }
            )
        except ImportError:
            
            from bayesian_network import BayesianNetwork
            
            bn = BayesianNetwork()
            
            evidence = {
                'Mode': 'future',
                'Time_Of_Day': 'peak',
                'Station_Type': 'interchange'
            }
            
            result = bn.infer(evidence, 'Crowding_Risk')
            
            print(f"Evidence: {evidence}")
            print(f"\nP(Crowding_Risk | evidence):")
            for risk in ['high', 'medium', 'low']:
                prob = result[risk]
                bar_length = int(prob * 50)
                bar = '█' * bar_length
                print(f"  {risk:8s} | {bar:50s} {prob:.2%}")
        
        input("\nPress Enter to continue...")


def requirement_4_optimization():
    """Q4: Optimization"""
    print_header("ADVANCED REQUIREMENT: OPTIMIZATION")
    
    print("\n\nMenu:")
    print("  1. Quick example")
    print("  2. Interactive optimization")
    print("  3. Back to main menu")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        # Quick example
        print_section("Quick Example: Disruption Re-routing")
        
        from optimization import (
            DisruptionScenario, OptimizationProblem, 
            SimulatedAnnealingOptimizer, MRTNetwork
        )
        
        network = MRTNetwork(
            "data/nodes.csv",
            "data/edges.csv",
            mode="future"
        )
        
        scenario = DisruptionScenario(
            "Tanah Merah-Expo Suspended",
            [("EW4", "CG1/DT35")],
            []
        )
        
        od_pairs = [
            ("CG2", "EW12/DT14"),
        ]
        
        problem = OptimizationProblem(network, od_pairs, scenario)
        
        print(f"Disruption: {scenario.name}")
        print(f"Baseline cost: {problem.baseline_cost:.2f} min\n")
        
        optimizer = SimulatedAnnealingOptimizer(problem)
        solution, cost, steps = optimizer.optimize(
            max_iterations=50,
            show_steps=True
        )
        
        input("\nPress Enter to continue...")
        
    elif choice == '2':
        # Interactive
        app = InteractiveOptimization(
            "data/nodes.csv",
            "data/edges.csv"
        )
        app.run()


def main():
    """Main application"""
    print_header("ChangiLink AI")

    
    while True:
        print("\n\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        
        print("\n  1. Search Algorithms (BFS, DFS, GBFS, A*)")
        print("  2. Logical Inference (Resolution)")
        print("  3. Bayesian Network (Crowding Risk)")
        print("  4. Optimization (Simulated Annealing)")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            requirement_1_search_algorithms()
        elif choice == '2':
            requirement_2_logical_inference()
        elif choice == '3':
            requirement_3_bayesian_network()
        elif choice == '4':
            requirement_4_optimization()
        elif choice == '5':
            print("\n" + "="*80)
            print("Thank you for using ChangiLink AI!")
            print("="*80 + "\n")
            break
        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()