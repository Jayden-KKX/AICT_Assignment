"""
ChangiLink AI - Demo Script
"""

import sys
from mrt_network import MRTNetwork
from search_algorithms import MRTRouteSearch, format_comparison_table
from logical_inference import initialize_mrt_rules, Clause
from bayesian_network import BayesianNetwork
from optimization import DisruptionScenario, OptimizationProblem, SimulatedAnnealingOptimizer
import time


def print_header(title: str):
    """Section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def print_subheader(title: str):
    """Subsection header"""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def run_search_experiments():
    """search algorithm  (Q1)"""
    
    print_header("Q1: ROUTE PLANNING WITH SEARCH ALGORITHMS")
    
    # Define test cases 
    test_cases_today = [
        ("CG2", "EW15", "Changi Airport → City Hall"),
        ("CG2", "DT19", "Changi Airport → Dhoby Ghaut (Orchard area)"),
        ("CG2", "TE22", "Changi Airport → Gardens by the Bay"),
        ("EW10", "CG2", "Paya Lebar → Changi Airport"),
    ]
    
    test_cases_future = [
        ("TE32/CR1", "EW15", "Changi Terminal 5 → City Hall"),
        ("TE32/CR1", "DT19", "Changi Terminal 5 → Dhoby Ghaut"),
        ("TE32/CR1", "TE22", "Changi Terminal 5 → Gardens by the Bay"),
        ("EW10", "TE32/CR1", "Paya Lebar → Changi Terminal 5"),
        ("DT32", "TE32/CR1", "Tampines East → Changi Terminal 5"),
    ]
    
    # Today Mode Experiments
    print_subheader("TODAY MODE - Current Network Operations")
    network_today = MRTNetwork(mode="today")
    searcher_today = MRTRouteSearch(network_today)
    
    for origin, dest, description in test_cases_today:
        print(f"\nTest Case: {description}")
        print(f"Origin: {origin} ({network_today.get_station(origin)})")
        print(f"Destination: {dest} ({network_today.get_station(dest)})\n")
        
        results = searcher_today.compare_all_algorithms(origin, dest)
        
        # Print individual results
        for algo_name, result in results.items():
            if result.path:
                print(f"{algo_name}:")
                print(f"  Path: {' → '.join([network_today.get_station(s).name for s in result.path[:5]])}{'...' if len(result.path) > 5 else ''}")
                print(f"  Stations: {len(result.path)}, Cost: {result.cost:.2f} min, Nodes expanded: {result.nodes_expanded}")
        
        print(f"\n{format_comparison_table(results)}")
        print("\n" + "~" * 80)
    
    # Future Mode Experiments
    print_subheader("FUTURE MODE - TELe/CRL Network w Terminal 5")
    network_future = MRTNetwork(mode="future")
    searcher_future = MRTRouteSearch(network_future)
    
    for origin, dest, description in test_cases_future:
        print(f"\nTest Case: {description}")
        print(f"Origin: {origin} ({network_future.get_station(origin)})")
        print(f"Destination: {dest} ({network_future.get_station(dest)})\n")
        
        results = searcher_future.compare_all_algorithms(origin, dest)
        
        # Print individual results
        for algo_name, result in results.items():
            if result.path:
                print(f"{algo_name}:")
                print(f"  Path: {' → '.join([network_future.get_station(s).name for s in result.path[:5]])}{'...' if len(result.path) > 5 else ''}")
                print(f"  Stations: {len(result.path)}, Cost: {result.cost:.2f} min, Nodes expanded: {result.nodes_expanded}")
        
        print(f"\n{format_comparison_table(results)}")
        print("\n" + "~" * 80)


def run_logical_inference_experiments():
    """Run logical inference experiments (Requirement 2)"""
    
    print_header("REQUIREMENT 2: LOGICAL INFERENCE FOR SERVICE RULES")
    
    # Scenario 1: Validate route under normal operations
    print_subheader("Scenario 1: Normal Operations - Route Validation")
    engine_today = initialize_mrt_rules(mode="today")
    
    route1 = ["CG2", "EW5", "EW4", "EW6", "EW8"]
    advisory1 = set()
    
    valid, msg = engine_today.is_route_valid(route1, advisory1)
    print(f"Route: {' → '.join(route1)}")
    print(f"Advisory: Normal operations (no disruptions)")
    print(f"Valid: {valid}")
    print(f"Explanation: {msg}")
    
    # Scenario 2: Disrupted segment
    print_subheader("Scenario 2: Service Disruption - Segment Closed")
    advisory2 = {
        Clause({"¬service_EW5_CG2"}),  # Expo to Changi Airport closed
    }
    
    route2 = ["EW5", "CG2"]
    valid, msg = engine_today.is_route_valid(route2, advisory2)
    print(f"Route: {' → '.join(route2)}")
    print(f"Advisory: Expo-Changi Airport segment suspended")
    print(f"Valid: {valid}")
    print(f"Explanation: {msg}")
    
    # Scenario 3: Systems integration work
    print_subheader("Scenario 3: Systems Integration at Tanah Merah")
    advisory3 = {
        Clause({"systems_integration"}),
        Clause({"¬service_EW4_EW5"}),  # Tanah Merah-Expo affected
    }
    
    consistent, msg = engine_today.is_consistent(advisory3)
    print(f"Advisory: Systems integration work at Tanah Merah-Expo corridor")
    print(f"Internally consistent: {consistent}")
    print(f"Explanation: {msg}")
    
    # Scenario 4: Future mode - TELe operational
    print_subheader("Scenario 4: Future Mode - TELe Operational")
    engine_future = initialize_mrt_rules(mode="future")
    
    query = Clause({"changi_via_tel_available"})
    result, explanation = engine_future.query(query)
    print(f"Query: Is Changi Airport accessible via TEL?")
    print(f"Result: {result}")
    print(f"Explanation: {explanation}")
    
    # Scenario 5: Contradictory advisories
    print_subheader("Scenario 5: Detect Contradictory Advisories")
    advisory5 = {
        Clause({"service_EW4_EW6"}),    # Service available Tanah Merah-Bedok
        Clause({"¬service_EW4_EW6"}),  # No service Tanah Merah-Bedok
    }
    
    consistent, msg = engine_today.is_consistent(advisory5)
    print(f"Advisory 1: Tanah Merah-Bedok service operating")
    print(f"Advisory 2: Tanah Merah-Bedok service suspended")
    print(f"Internally consistent: {consistent}")
    print(f"Explanation: {msg}")


def run_bayesian_network_experiments():
    """Run Bayesian network experiments (Requirement 3)"""
    
    print_header("REQUIREMENT 3: BAYESIAN NETWORK FOR CROWDING RISK")
    
    bn = BayesianNetwork()
    
    # Scenario 1: Today mode, peak, interchange
    print_subheader("Scenario 1: Today Mode - Peak Hours at Interchange")
    evidence1 = {
        'Mode': 'today',
        'Time_Of_Day': 'peak',
        'Station_Type': 'interchange'
    }
    result1 = bn.infer(evidence1, 'Crowding_Risk')
    print(f"Evidence: {evidence1}")
    print(f"P(Crowding_Risk | evidence):")
    for risk in ['high', 'medium', 'low']:
        print(f"  {risk}: {result1[risk]:.3f} ({result1[risk]*100:.1f}%)")
    
    # Scenario 2: Future mode, peak, interchange
    print_subheader("Scenario 2: Future Mode - Peak Hours at Interchange")
    evidence2 = {
        'Mode': 'future',
        'Time_Of_Day': 'peak',
        'Station_Type': 'interchange'
    }
    result2 = bn.infer(evidence2, 'Crowding_Risk')
    print(f"Evidence: {evidence2}")
    print(f"P(Crowding_Risk | evidence):")
    for risk in ['high', 'medium', 'low']:
        print(f"  {risk}: {result2[risk]:.3f} ({result2[risk]*100:.1f}%)")
    
    print(f"\n** COMPARISON (Today vs Future at peak interchange) **")
    print(f"High risk: {result1['high']:.3f} → {result2['high']:.3f} (change: {result2['high']-result1['high']:+.3f})")
    print(f"\nExplanation:")
    print(f"Future mode introduces major network changes (TELe/CRL extension), leading to")
    print(f"concentrated passenger flow patterns as commuters adjust to new routes. This")
    print(f"increases crowding risk at key interchanges during peak hours.")
    
    # Scenario 3: Today mode, off-peak, airport
    print_subheader("Scenario 3: Today Mode - Off-Peak at Airport Station")
    evidence3 = {
        'Mode': 'today',
        'Time_Of_Day': 'off_peak',
        'Station_Type': 'airport'
    }
    result3 = bn.infer(evidence3, 'Crowding_Risk')
    print(f"Evidence: {evidence3}")
    print(f"P(Crowding_Risk | evidence):")
    for risk in ['high', 'medium', 'low']:
        print(f"  {risk}: {result3[risk]:.3f} ({result3[risk]*100:.1f}%)")
    
    # Scenario 4: Future mode, off-peak, airport
    print_subheader("Scenario 4: Future Mode - Off-Peak at Airport Station")
    evidence4 = {
        'Mode': 'future',
        'Time_Of_Day': 'off_peak',
        'Station_Type': 'airport'
    }
    result4 = bn.infer(evidence4, 'Crowding_Risk')
    print(f"Evidence: {evidence4}")
    print(f"P(Crowding_Risk | evidence):")
    for risk in ['high', 'medium', 'low']:
        print(f"  {risk}: {result4[risk]:.3f} ({result4[risk]*100:.1f}%)")
    
    print(f"\n** COMPARISON (Today vs Future at off-peak airport) **")
    print(f"High risk: {result3['high']:.3f} → {result4['high']:.3f} (change: {result4['high']-result3['high']:+.3f})")
    print(f"\nExplanation:")
    print(f"T5 availability in Future mode provides better distribution of passenger flow")
    print(f"across multiple airport stations. Even with major network changes, the additional")
    print(f"capacity helps reduce concentrated crowding during off-peak periods.")
    
    # Scenario 5: Today mode, peak, regular
    print_subheader("Scenario 5: Today Mode - Peak Hours at Regular Station")
    evidence5 = {
        'Mode': 'today',
        'Time_Of_Day': 'peak',
        'Station_Type': 'regular'
    }
    result5 = bn.infer(evidence5, 'Crowding_Risk')
    print(f"Evidence: {evidence5}")
    print(f"P(Crowding_Risk | evidence):")
    for risk in ['high', 'medium', 'low']:
        print(f"  {risk}: {result5[risk]:.3f} ({result5[risk]*100:.1f}%)")


def run_optimization_experiments():
    """Run optimization experiments (Advanced Requirement)"""
    
    print_header("ADVANCED REQUIREMENT: OPTIMIZATION FOR DISRUPTION RE-ROUTING")
    
    network = MRTNetwork(mode="future")
    
    # Disruption Scenario 1
    print_subheader("Disruption Scenario 1: Tanah Merah-Expo Segment Suspended")
    
    disruption1 = DisruptionScenario(
        "Tanah Merah-Expo Suspended + Expo-Changi Reduced",
        disrupted_segments=[("EW4", "EW5")],
        reduced_frequency_segments=[("EW5", "CG2")]
    )
    
    od_pairs1 = [
        ("CG2", "EW15"),      # Changi Airport to City Hall
        ("TE32/CR1", "TE22"), # T5 to Gardens by the Bay
        ("EW10", "CG2"),      # Paya Lebar to Changi Airport
        ("TE31", "EW18"),     # Sungei Bedok to Outram Park
    ]
    
    problem1 = OptimizationProblem(network, od_pairs1, disruption1, max_transfers=3)
    
    print(f"OD Pairs: {len(od_pairs1)}")
    print(f"Disrupted: Tanah Merah ↔ Expo")
    print(f"Reduced Frequency: Expo ↔ Changi Airport")
    print(f"Baseline total travel time: {problem1.baseline_cost:.2f} minutes")
    
    # Run Simulated Annealing
    import random
    random.seed(42)
    
    sa_optimizer = SimulatedAnnealingOptimizer(problem1)
    sa_solution, sa_cost, sa_history = sa_optimizer.optimize(
        max_iterations=100, 
        initial_temp=50.0, 
        cooling_rate=0.95
    )
    
    print(f"\nOptimized total travel time: {sa_cost:.2f} minutes")
    print(f"Improvement: {problem1.baseline_cost - sa_cost:.2f} minutes")
    print(f"Percentage improvement: {((problem1.baseline_cost - sa_cost) / problem1.baseline_cost * 100):.1f}%")
    
    # Show example routes
    print(f"\nExample Re-routed Path:")
    od = od_pairs1[0]
    baseline_path = problem1.baseline_routes.get(od, [])
    optimized_path = sa_solution.get(od, [])
    
    print(f"OD: {od[0]} → {od[1]}")
    print(f"Baseline:  {' → '.join([network.get_station(s).name for s in baseline_path][:7])}...")
    print(f"Optimized: {' → '.join([network.get_station(s).name for s in optimized_path][:7])}...")


def main():
    """Run all experiments"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CHANGILINK AI - Demo Script".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "Singapore MRT Network with TELe/CRL Extension".center(78) + "║")
    print("║" + "Based on LTA's July 2025 Announcement".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    start_time = time.time()
    
    try:
        # Run all experiments
        run_search_experiments()
        run_logical_inference_experiments()
        run_bayesian_network_experiments()
        run_optimization_experiments()
        
        
    except Exception as e:
        print(f"\n Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
