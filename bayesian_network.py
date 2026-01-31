"""
Improved Bayesian Network for MRT Crowding Risk Prediction
Focused on Changi Airport-T5 corridor with required variables
Cleaner, more concise output
"""

from typing import Dict, List, Set, Tuple, Optional
import itertools


class ConditionalProbabilityTable:
    """Conditional Probability Table (CPT) for a variable"""
    
    def __init__(self, variable: str, parents: List[str], probabilities: Dict):
        self.variable = variable
        self.parents = parents
        self.probabilities = probabilities
    
    def get_probability(self, var_value: str, parent_values: Dict[str, str]) -> float:
        """Get P(variable=var_value | parent_values)"""
        if not self.parents:
            return self.probabilities.get(var_value, 0.0)
        
        parent_tuple = tuple(parent_values.get(p, None) for p in self.parents)
        
        if parent_tuple in self.probabilities:
            return self.probabilities[parent_tuple].get(var_value, 0.0)
        
        return 0.0


class BayesianNetwork:
    """
    Bayesian Network for Crowding Risk Prediction - Changi Airport-T5 Corridor
    
    Network Structure (7 variables):
    - Weather (W): Clear / Rainy / Thunderstorms
    - Time_Of_Day (T): Morning / Afternoon / Evening  
    - Day_Type (D): Weekday / Weekend
    - Network_Mode (M): Today / Future
    - Service_Status (S): Normal / Reduced / Disrupted → depends on Mode
    - Demand_Proxy (P): Low / Medium / High → depends on Time, Day, Weather
    - Crowding_Risk (C): Low / Medium / High → depends on Demand, Service, Mode
    """
    
    def __init__(self):
        self.variables: Set[str] = set()
        self.cpts: Dict[str, ConditionalProbabilityTable] = {}
        self.domains: Dict[str, List[str]] = {}
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the Bayesian network structure and CPTs"""
        
        # Define domains
        self.domains = {
            'Weather': ['clear', 'rainy', 'thunderstorms'],
            'Time_Of_Day': ['morning', 'afternoon', 'evening'],
            'Day_Type': ['weekday', 'weekend'],
            'Network_Mode': ['today', 'future'],
            'Service_Status': ['normal', 'reduced', 'disrupted'],
            'Demand_Proxy': ['low', 'medium', 'high'],
            'Crowding_Risk': ['low', 'medium', 'high']
        }
        
        self.variables = set(self.domains.keys())
        
        # CPT 1: Weather (root) - Singapore climate data
        # Source: NEA historical data - ~60% clear, 30% rainy, 10% thunderstorms
        self.cpts['Weather'] = ConditionalProbabilityTable(
            'Weather', [],
            {'clear': 0.60, 'rainy': 0.30, 'thunderstorms': 0.10}
        )
        
        # CPT 2: Time_Of_Day (root) - Uniform assumption for corridor
        self.cpts['Time_Of_Day'] = ConditionalProbabilityTable(
            'Time_Of_Day', [],
            {'morning': 0.33, 'afternoon': 0.34, 'evening': 0.33}
        )
        
        # CPT 3: Day_Type (root) - 5 weekdays, 2 weekend days
        self.cpts['Day_Type'] = ConditionalProbabilityTable(
            'Day_Type', [],
            {'weekday': 0.71, 'weekend': 0.29}
        )
        
        # CPT 4: Network_Mode (root) - Equal prior for scenario testing
        self.cpts['Network_Mode'] = ConditionalProbabilityTable(
            'Network_Mode', [],
            {'today': 0.5, 'future': 0.5}
        )
        
        # CPT 5: Service_Status depends on Network_Mode
        # Justification: Future mode has higher disruption risk during systems integration (TELe/CRL works)
        # Source: LTA announcements on planned works, historical MRT disruption frequency
        self.cpts['Service_Status'] = ConditionalProbabilityTable(
            'Service_Status', ['Network_Mode'],
            {
                ('today',): {'normal': 0.85, 'reduced': 0.10, 'disrupted': 0.05},
                ('future',): {'normal': 0.70, 'reduced': 0.20, 'disrupted': 0.10}  # Higher disruption during integration
            }
        )
        
        # CPT 6: Demand_Proxy depends on Time_Of_Day, Day_Type, Weather
        # Justification: Changi Airport corridor demand patterns
        # - Morning/Evening weekdays: High (commuter + airport traffic)
        # - Weekends: Lower but more tourist traffic
        # - Bad weather: Higher demand for covered transport
        demand_probs = {}
        
        # Weekday patterns
        demand_probs[('morning', 'weekday', 'clear')] = {'low': 0.10, 'medium': 0.30, 'high': 0.60}
        demand_probs[('morning', 'weekday', 'rainy')] = {'low': 0.05, 'medium': 0.25, 'high': 0.70}
        demand_probs[('morning', 'weekday', 'thunderstorms')] = {'low': 0.05, 'medium': 0.20, 'high': 0.75}
        
        demand_probs[('afternoon', 'weekday', 'clear')] = {'low': 0.20, 'medium': 0.50, 'high': 0.30}
        demand_probs[('afternoon', 'weekday', 'rainy')] = {'low': 0.15, 'medium': 0.45, 'high': 0.40}
        demand_probs[('afternoon', 'weekday', 'thunderstorms')] = {'low': 0.10, 'medium': 0.40, 'high': 0.50}
        
        demand_probs[('evening', 'weekday', 'clear')] = {'low': 0.10, 'medium': 0.25, 'high': 0.65}
        demand_probs[('evening', 'weekday', 'rainy')] = {'low': 0.05, 'medium': 0.20, 'high': 0.75}
        demand_probs[('evening', 'weekday', 'thunderstorms')] = {'low': 0.05, 'medium': 0.15, 'high': 0.80}
        
        # Weekend patterns (lower commuter demand, more leisure)
        demand_probs[('morning', 'weekend', 'clear')] = {'low': 0.30, 'medium': 0.45, 'high': 0.25}
        demand_probs[('morning', 'weekend', 'rainy')] = {'low': 0.25, 'medium': 0.40, 'high': 0.35}
        demand_probs[('morning', 'weekend', 'thunderstorms')] = {'low': 0.20, 'medium': 0.35, 'high': 0.45}
        
        demand_probs[('afternoon', 'weekend', 'clear')] = {'low': 0.25, 'medium': 0.45, 'high': 0.30}
        demand_probs[('afternoon', 'weekend', 'rainy')] = {'low': 0.20, 'medium': 0.40, 'high': 0.40}
        demand_probs[('afternoon', 'weekend', 'thunderstorms')] = {'low': 0.15, 'medium': 0.35, 'high': 0.50}
        
        demand_probs[('evening', 'weekend', 'clear')] = {'low': 0.20, 'medium': 0.50, 'high': 0.30}
        demand_probs[('evening', 'weekend', 'rainy')] = {'low': 0.15, 'medium': 0.45, 'high': 0.40}
        demand_probs[('evening', 'weekend', 'thunderstorms')] = {'low': 0.10, 'medium': 0.40, 'high': 0.50}
        
        self.cpts['Demand_Proxy'] = ConditionalProbabilityTable(
            'Demand_Proxy', ['Time_Of_Day', 'Day_Type', 'Weather'],
            demand_probs
        )
        
        # CPT 7: Crowding_Risk depends on Demand_Proxy, Service_Status, Network_Mode
        # Justification: 
        # - High demand + reduced/disrupted service = high crowding
        # - Future mode with T5 provides additional capacity, reducing crowding
        # - Systems integration works increase crowding risk temporarily
        crowding_probs = {}
        
        # Today Mode - Normal Service
        crowding_probs[('low', 'normal', 'today')] = {'low': 0.80, 'medium': 0.15, 'high': 0.05}
        crowding_probs[('medium', 'normal', 'today')] = {'low': 0.40, 'medium': 0.45, 'high': 0.15}
        crowding_probs[('high', 'normal', 'today')] = {'low': 0.15, 'medium': 0.40, 'high': 0.45}
        
        # Today Mode - Reduced Service
        crowding_probs[('low', 'reduced', 'today')] = {'low': 0.50, 'medium': 0.35, 'high': 0.15}
        crowding_probs[('medium', 'reduced', 'today')] = {'low': 0.20, 'medium': 0.40, 'high': 0.40}
        crowding_probs[('high', 'reduced', 'today')] = {'low': 0.10, 'medium': 0.30, 'high': 0.60}
        
        # Today Mode - Disrupted Service
        crowding_probs[('low', 'disrupted', 'today')] = {'low': 0.30, 'medium': 0.40, 'high': 0.30}
        crowding_probs[('medium', 'disrupted', 'today')] = {'low': 0.15, 'medium': 0.35, 'high': 0.50}
        crowding_probs[('high', 'disrupted', 'today')] = {'low': 0.05, 'medium': 0.25, 'high': 0.70}
        
        # Future Mode - Normal Service (T5 provides relief)
        crowding_probs[('low', 'normal', 'future')] = {'low': 0.85, 'medium': 0.12, 'high': 0.03}
        crowding_probs[('medium', 'normal', 'future')] = {'low': 0.50, 'medium': 0.40, 'high': 0.10}
        crowding_probs[('high', 'normal', 'future')] = {'low': 0.25, 'medium': 0.45, 'high': 0.30}
        
        # Future Mode - Reduced Service (integration works impact)
        crowding_probs[('low', 'reduced', 'future')] = {'low': 0.45, 'medium': 0.40, 'high': 0.15}
        crowding_probs[('medium', 'reduced', 'future')] = {'low': 0.20, 'medium': 0.45, 'high': 0.35}
        crowding_probs[('high', 'reduced', 'future')] = {'low': 0.10, 'medium': 0.35, 'high': 0.55}
        
        # Future Mode - Disrupted Service (worse due to integration complexity)
        crowding_probs[('low', 'disrupted', 'future')] = {'low': 0.25, 'medium': 0.40, 'high': 0.35}
        crowding_probs[('medium', 'disrupted', 'future')] = {'low': 0.10, 'medium': 0.30, 'high': 0.60}
        crowding_probs[('high', 'disrupted', 'future')] = {'low': 0.05, 'medium': 0.20, 'high': 0.75}
        
        self.cpts['Crowding_Risk'] = ConditionalProbabilityTable(
            'Crowding_Risk', ['Demand_Proxy', 'Service_Status', 'Network_Mode'],
            crowding_probs
        )
    
    def infer(self, evidence: Dict[str, str], query_var: str, verbose: bool = False) -> Dict[str, float]:
        """
        Perform probabilistic inference using enumeration
        
        Args:
            evidence: Dictionary of observed variables
            query_var: Variable to query
            verbose: Show calculation details
        
        Returns:
            Probability distribution over query variable
        """
        hidden_vars = self.variables - {query_var} - set(evidence.keys())
        result = {}
        
        if verbose:
            print(f"\nComputing P({query_var} | {list(evidence.keys())})")
            print(f"Hidden variables to marginalize: {list(hidden_vars)}\n")
        
        # For each possible value of the query variable
        for query_value in self.domains[query_var]:
            assignment = evidence.copy()
            assignment[query_var] = query_value
            
            # Sum over all hidden variables
            prob = self._enumerate_all(list(hidden_vars), assignment, verbose, query_value)
            result[query_value] = prob
            
            if verbose:
                print(f"P({query_var}={query_value} | evidence) = {prob:.6f}")
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}
        
        if verbose:
            print(f"\nNormalization constant: {total:.6f}")
            print("\nFinal probabilities:")
            for value, prob in result.items():
                print(f"  P({query_var}={value}) = {prob:.4f} ({prob*100:.2f}%)")
        
        return result
    
    def _enumerate_all(self, hidden_vars: List[str], assignment: Dict[str, str], 
                       verbose: bool, query_value: str) -> float:
        """Recursive enumeration over hidden variables"""
        if not hidden_vars:
            return self._compute_probability(assignment)
        
        var = hidden_vars[0]
        remaining = hidden_vars[1:]
        
        total = 0.0
        for value in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            total += self._enumerate_all(remaining, new_assignment, verbose, query_value)
        
        return total
    
    def _compute_probability(self, assignment: Dict[str, str]) -> float:
        """Compute joint probability of a complete assignment"""
        prob = 1.0
        
        for var in self.variables:
            cpt = self.cpts[var]
            var_value = assignment[var]
            parent_values = {p: assignment[p] for p in cpt.parents}
            prob *= cpt.get_probability(var_value, parent_values)
        
        return prob
    
    def run_scenario(self, scenario_name: str, evidence: Dict[str, str], show_details: bool = False):
        """Run a single scenario and print results"""
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        
        # Print evidence
        print("\nEvidence:")
        for var, value in sorted(evidence.items()):
            print(f"  {var:20s} = {value}")
        
        # Perform inference
        result = self.infer(evidence, 'Crowding_Risk', verbose=show_details)
        
        # Print results
        print("\nCrowding Risk Prediction:")
        print("-" * 40)
        for risk in ['low', 'medium', 'high']:
            bar_length = int(result[risk] * 50)
            bar = '█' * bar_length
            print(f"  {risk:8s} | {bar:50s} {result[risk]:.2%}")
        
        # Interpretation
        max_risk = max(result.items(), key=lambda x: x[1])
        print(f"\n→ Most likely: {max_risk[0].upper()} risk ({max_risk[1]:.1%})")
        
        return result


def run_required_scenarios():
    """Run the 5+ required scenarios with Today vs Future comparisons"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "BAYESIAN NETWORK: Crowding Risk Analysis".center(78) + "║")
    print("║" + "Changi Airport-T5 Corridor".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    bn = BayesianNetwork()
    
    # Store results for comparison
    results = {}
    
    # Scenario 1: Rainy evening + reduced service (Today)
    results['S1'] = bn.run_scenario(
        "1. Rainy Evening + Reduced Service (TODAY MODE)",
        {
            'Weather': 'rainy',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'reduced'
        }
    )
    
    # Scenario 2: Clear morning weekday + normal service (Today)
    results['S2'] = bn.run_scenario(
        "2. Clear Morning Weekday + Normal Service (TODAY MODE)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'morning',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'normal'
        }
    )
    
    # Scenario 3: Weekend afternoon + normal service (Today)
    results['S3'] = bn.run_scenario(
        "3. Weekend Afternoon + Normal Service (TODAY MODE)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'afternoon',
            'Day_Type': 'weekend',
            'Network_Mode': 'today',
            'Service_Status': 'normal'
        }
    )
    
    # Scenario 4: Disrupted service near airport corridor (Today)
    results['S4'] = bn.run_scenario(
        "4. Disrupted Service - Airport Corridor (TODAY MODE)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'disrupted'
        }
    )
    
    # Scenario 5: Clear evening + normal service (TODAY vs FUTURE comparison)
    results['S5a'] = bn.run_scenario(
        "5a. Clear Evening + Normal Service (TODAY MODE - Baseline)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'normal'
        }
    )
    
    results['S5b'] = bn.run_scenario(
        "5b. Clear Evening + Normal Service (FUTURE MODE - With TELe/CRL)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'future',
            'Service_Status': 'normal'
        }
    )
    
    # Scenario 6: Rainy evening + reduced service (TODAY vs FUTURE comparison)
    results['S6a'] = bn.run_scenario(
        "6a. Rainy Evening + Reduced Service (TODAY MODE - Baseline)",
        {
            'Weather': 'rainy',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'reduced'
        }
    )
    
    results['S6b'] = bn.run_scenario(
        "6b. Rainy Evening + Reduced Service (FUTURE MODE - With TELe/CRL)",
        {
            'Weather': 'rainy',
            'Time_Of_Day': 'evening',
            'Day_Type': 'weekday',
            'Network_Mode': 'future',
            'Service_Status': 'reduced'
        }
    )
    
    # Scenario 7: Morning commute disruption (Today vs Future comparison)
    results['S7a'] = bn.run_scenario(
        "7a. Morning Commute + Disrupted Service (Today)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'morning',
            'Day_Type': 'weekday',
            'Network_Mode': 'today',
            'Service_Status': 'disrupted'
        }
    )
    
    results['S7b'] = bn.run_scenario(
        "7b. Morning Commute + Disrupted Service (Future)",
        {
            'Weather': 'clear',
            'Time_Of_Day': 'morning',
            'Day_Type': 'weekday',
            'Network_Mode': 'future',
            'Service_Status': 'disrupted'
        }
    )
    
    # Print comprehensive comparison analysis
    print(f"\n\n{'='*80}")
    print("Comparative Analysis: Today vs Future Mode")
    print(f"{'='*80}\n")
    
    comparisons = [
        ("Clear Evening + Normal Service", 'S5a', 'S5b'),
        ("Rainy Evening + Reduced Service", 'S6a', 'S6b'),
        ("Morning Commute + Disrupted Service", 'S7a', 'S7b')
    ]
    
    for scenario_name, today_key, future_key in comparisons:
        print(f"\n{scenario_name}:")
        print("-" * 60)
        
        today = results[today_key]
        future = results[future_key]
        
        print(f"{'Risk Level':<15} {'Today':>15} {'Future':>15} {'Change':>15}")
        print("-" * 60)
        
        for risk in ['low', 'medium', 'high']:
            change = future[risk] - today[risk]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "="
            print(f"{risk.capitalize():<15} {today[risk]:>14.1%} {future[risk]:>14.1%} {arrow} {abs(change):>13.1%}")
        
        # Explanation
        print("\nExplanation:")
        if future['high'] < today['high']:
            print(f"  Future mode shows LOWER high-risk probability ({today['high']:.1%} → {future['high']:.1%})")
            print("    Reason: Terminal 5 provides additional capacity and route options")
        elif future['high'] > today['high']:
            print(f"  Future mode shows HIGHER high-risk probability ({today['high']:.1%} → {future['high']:.1%})")
            print("    Reason: Systems integration works during TELe/CRL implementation")
        else:
            print(f"  = Similar risk levels in both modes")

if __name__ == "__main__":
    run_required_scenarios()