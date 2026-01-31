"""
Enhanced Bayesian Network for MRT Crowding Risk Prediction
Implements Variable Elimination with step-by-step workings
Interactive scenario testing and probability visualization
"""

from typing import Dict, List, Set, Tuple, Optional
import itertools


class InferenceStep:
    """Records a step in the inference process"""
    def __init__(self, step_num: int, action: str, details: Dict):
        self.step_num = step_num
        self.action = action  # 'enumerate', 'compute_joint', 'marginalize', 'normalize'
        self.details = details
    
    def __repr__(self):
        return f"Step {self.step_num}: {self.action} - {self.details}"


class ConditionalProbabilityTable:
    """Conditional Probability Table (CPT) for a variable"""
    
    def __init__(self, variable: str, parents: List[str], probabilities: Dict):
        """
        Initialize CPT
        
        Args:
            variable: The variable name
            parents: List of parent variable names
            probabilities: Dict mapping parent values to probability distributions
        """
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
    
    def get_all_values(self, parent_values: Dict[str, str]) -> Dict[str, float]:
        """Get all probability values for this variable given parent values"""
        if not self.parents:
            return self.probabilities
        
        parent_tuple = tuple(parent_values.get(p, None) for p in self.parents)
        return self.probabilities.get(parent_tuple, {})
    
    def print_table(self):
        """Print the CPT in a readable format"""
        print(f"\n  CPT for {self.variable}:")
        if not self.parents:
            print(f"    (No parents - prior probability)")
            for value, prob in self.probabilities.items():
                print(f"      P({self.variable}={value}) = {prob:.3f}")
        else:
            print(f"    Parents: {', '.join(self.parents)}")
            for parent_config, probs in self.probabilities.items():
                parent_str = ', '.join([f"{p}={v}" for p, v in zip(self.parents, parent_config)])
                print(f"    Given {parent_str}:")
                for value, prob in probs.items():
                    print(f"      P({self.variable}={value}) = {prob:.3f}")


class BayesianNetwork:
    """
    Bayesian Network for crowding risk prediction
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
            'Mode': ['today', 'future'],
            'Time_Of_Day': ['peak', 'off_peak'],
            'Station_Type': ['interchange', 'regular', 'airport'],
            'T5_Available': ['yes', 'no'],
            'Network_Changes': ['major', 'minor', 'none'],
            'Passenger_Flow_Pattern': ['concentrated', 'distributed', 'normal'],
            'Crowding_Risk': ['high', 'medium', 'low']
        }
        
        self.variables = set(self.domains.keys())
        
        # CPT 1: Mode (root node)
        self.cpts['Mode'] = ConditionalProbabilityTable(
            'Mode', [],
            {'today': 0.5, 'future': 0.5}
        )
        
        # CPT 2: Time_Of_Day (root node)
        self.cpts['Time_Of_Day'] = ConditionalProbabilityTable(
            'Time_Of_Day', [],
            {'peak': 0.22, 'off_peak': 0.78}
        )
        
        # CPT 3: Station_Type (root node)
        self.cpts['Station_Type'] = ConditionalProbabilityTable(
            'Station_Type', [],
            {'interchange': 0.20, 'airport': 0.10, 'regular': 0.70}
        )
        
        # CPT 4: T5_Available depends on Mode
        self.cpts['T5_Available'] = ConditionalProbabilityTable(
            'T5_Available', ['Mode'],
            {
                ('today',): {'yes': 0.0, 'no': 1.0},
                ('future',): {'yes': 1.0, 'no': 0.0}
            }
        )
        
        # CPT 5: Network_Changes depends on Mode
        self.cpts['Network_Changes'] = ConditionalProbabilityTable(
            'Network_Changes', ['Mode'],
            {
                ('today',): {'major': 0.0, 'minor': 0.1, 'none': 0.9},
                ('future',): {'major': 0.8, 'minor': 0.15, 'none': 0.05}
            }
        )
        
        # CPT 6: Passenger_Flow_Pattern depends on T5_Available and Network_Changes
        self.cpts['Passenger_Flow_Pattern'] = ConditionalProbabilityTable(
            'Passenger_Flow_Pattern', ['T5_Available', 'Network_Changes'],
            {
                ('yes', 'major'): {'concentrated': 0.5, 'distributed': 0.3, 'normal': 0.2},
                ('yes', 'minor'): {'concentrated': 0.3, 'distributed': 0.4, 'normal': 0.3},
                ('yes', 'none'): {'concentrated': 0.2, 'distributed': 0.3, 'normal': 0.5},
                ('no', 'major'): {'concentrated': 0.6, 'distributed': 0.2, 'normal': 0.2},
                ('no', 'minor'): {'concentrated': 0.3, 'distributed': 0.3, 'normal': 0.4},
                ('no', 'none'): {'concentrated': 0.1, 'distributed': 0.2, 'normal': 0.7},
            }
        )
        
        # CPT 7: Crowding_Risk depends on Time_Of_Day, Station_Type, and Passenger_Flow_Pattern
        crowding_probs = {}
        
        # Peak hours scenarios
        crowding_probs[('peak', 'interchange', 'concentrated')] = {'high': 0.8, 'medium': 0.15, 'low': 0.05}
        crowding_probs[('peak', 'interchange', 'distributed')] = {'high': 0.5, 'medium': 0.35, 'low': 0.15}
        crowding_probs[('peak', 'interchange', 'normal')] = {'high': 0.6, 'medium': 0.3, 'low': 0.1}
        
        crowding_probs[('peak', 'airport', 'concentrated')] = {'high': 0.7, 'medium': 0.2, 'low': 0.1}
        crowding_probs[('peak', 'airport', 'distributed')] = {'high': 0.4, 'medium': 0.4, 'low': 0.2}
        crowding_probs[('peak', 'airport', 'normal')] = {'high': 0.5, 'medium': 0.35, 'low': 0.15}
        
        crowding_probs[('peak', 'regular', 'concentrated')] = {'high': 0.5, 'medium': 0.3, 'low': 0.2}
        crowding_probs[('peak', 'regular', 'distributed')] = {'high': 0.3, 'medium': 0.4, 'low': 0.3}
        crowding_probs[('peak', 'regular', 'normal')] = {'high': 0.4, 'medium': 0.4, 'low': 0.2}
        
        # Off-peak scenarios
        crowding_probs[('off_peak', 'interchange', 'concentrated')] = {'high': 0.4, 'medium': 0.4, 'low': 0.2}
        crowding_probs[('off_peak', 'interchange', 'distributed')] = {'high': 0.2, 'medium': 0.4, 'low': 0.4}
        crowding_probs[('off_peak', 'interchange', 'normal')] = {'high': 0.3, 'medium': 0.4, 'low': 0.3}
        
        crowding_probs[('off_peak', 'airport', 'concentrated')] = {'high': 0.3, 'medium': 0.4, 'low': 0.3}
        crowding_probs[('off_peak', 'airport', 'distributed')] = {'high': 0.2, 'medium': 0.3, 'low': 0.5}
        crowding_probs[('off_peak', 'airport', 'normal')] = {'high': 0.25, 'medium': 0.35, 'low': 0.4}
        
        crowding_probs[('off_peak', 'regular', 'concentrated')] = {'high': 0.2, 'medium': 0.3, 'low': 0.5}
        crowding_probs[('off_peak', 'regular', 'distributed')] = {'high': 0.1, 'medium': 0.3, 'low': 0.6}
        crowding_probs[('off_peak', 'regular', 'normal')] = {'high': 0.15, 'medium': 0.35, 'low': 0.5}
        
        self.cpts['Crowding_Risk'] = ConditionalProbabilityTable(
            'Crowding_Risk', ['Time_Of_Day', 'Station_Type', 'Passenger_Flow_Pattern'],
            crowding_probs
        )
    
    def infer(self, evidence: Dict[str, str], query_var: str, show_steps: bool = False) -> Tuple[Dict[str, float], List[InferenceStep]]:
        """
        Perform probabilistic inference using enumeration
        
        Args:
            evidence: observed variables and their values
            query_var: variable to query
            show_steps: to show detailed computation steps
        
        Returns:
            Tuple of (probability distribution, inference steps)
        """
        hidden_vars = self.variables - {query_var} - set(evidence.keys())
        steps = []
        step_num = 0
        
        result = {}
        
        if show_steps:
            print(f"\n{'='*80}")
            print(f"BAYESIAN INFERENCE: Computing P({query_var} | {evidence})")
            print(f"{'='*80}\n")
            print(f"Evidence variables: {list(evidence.keys())}")
            print(f"Query variable: {query_var}")
            print(f"Hidden variables: {list(hidden_vars)}")
            print(f"\nNetwork structure:")
            for var, cpt in self.cpts.items():
                if cpt.parents:
                    print(f"  {var} ← {', '.join(cpt.parents)}")
                else:
                    print(f"  {var} (root node)")
        
        # For each possible value of the query variable
        for query_value in self.domains[query_var]:
            step_num += 1
            assignment = evidence.copy()
            assignment[query_var] = query_value
            
            if show_steps:
                print(f"\n{'-'*80}")
                print(f"Step {step_num}: Computing P({query_var}={query_value} | evidence)")
                print(f"{'-'*80}")
            
            # Sum over all possible assignments to hidden variables
            prob = self._enumerate_all(list(hidden_vars), assignment, show_steps, step_num)
            result[query_value] = prob
            
            if show_steps:
                print(f"\n  → P({query_var}={query_value} | evidence) = {prob:.6f}")
            
            steps.append(InferenceStep(step_num, 'enumerate', {
                'query_value': query_value,
                'unnormalized_prob': prob
            }))
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}
        
        if show_steps:
            print(f"\n{'='*80}")
            print("NORMALIZATION")
            print(f"{'='*80}")
            print(f"\nSum of unnormalized probabilities: {total:.6f}")
            print(f"\nNormalized probabilities:")
            for value, prob in result.items():
                print(f"  P({query_var}={value} | evidence) = {prob:.4f} ({prob*100:.2f}%)")
        
        steps.append(InferenceStep(step_num + 1, 'normalize', {
            'total': total,
            'result': result.copy()
        }))
        
        return result, steps
    
    def _enumerate_all(self, hidden_vars: List[str], assignment: Dict[str, str], 
                       show_steps: bool, base_step: int) -> float:
        """Recursive enumeration over all hidden variables"""
        if not hidden_vars:
            # Base case: compute probability of full assignment
            prob = self._compute_probability(assignment, show_steps)
            return prob
        
        # Recursive case: sum over values of first hidden variable
        var = hidden_vars[0]
        remaining = hidden_vars[1:]
        
        if show_steps:
            print(f"\n  Marginalizing over {var}:")
        
        total = 0.0
        for value in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            if show_steps:
                print(f"    {var}={value}:")
            
            prob = self._enumerate_all(remaining, new_assignment, show_steps, base_step)
            total += prob
            
            if show_steps:
                print(f"      → contributes {prob:.6f}")
        
        return total
    
    def _compute_probability(self, assignment: Dict[str, str], show_steps: bool = False) -> float:
        """Compute joint probability of a complete assignment"""
        prob = 1.0
        
        if show_steps:
            print(f"      Computing joint probability for:")
            for var in sorted(assignment.keys()):
                print(f"        {var}={assignment[var]}")
        
        for var in self.variables:
            cpt = self.cpts[var]
            var_value = assignment[var]
            parent_values = {p: assignment[p] for p in cpt.parents}
            
            p = cpt.get_probability(var_value, parent_values)
            prob *= p
            
            if show_steps:
                if cpt.parents:
                    parent_str = ', '.join([f"{p}={assignment[p]}" for p in cpt.parents])
                    print(f"        P({var}={var_value} | {parent_str}) = {p:.3f}")
                else:
                    print(f"        P({var}={var_value}) = {p:.3f}")
        
        if show_steps:
            print(f"      Joint probability = {prob:.6f}")
        
        return prob
    
    def print_network_structure(self):
        """Print the network structure and CPTs"""
        print(f"\n{'='*80}")
        print("Bayesian Network Structure")
        print(f"{'='*80}\n")
        
        print(f"Variables: {len(self.variables)}")
        for var in sorted(self.variables):
            print(f"  • {var}: {self.domains[var]}")
        
        print(f"\n\nConditional Probability Tables:")
        print("-" * 80)
        
        for var in sorted(self.variables):
            self.cpts[var].print_table()
    
    def compare_scenarios(self, scenarios: List[Tuple[str, Dict[str, str]]], query_var: str):
        """Compare multiple scenarios side-by-side"""
        print(f"\n{'='*80}")
        print(f"Scenario Comparison: P({query_var} | different evidence)")
        print(f"{'='*80}\n")
        
        results = []
        for name, evidence in scenarios:
            result, _ = self.infer(evidence, query_var, show_steps=False)
            results.append((name, evidence, result))
        
        # Print comparison table
        print(f"{'Scenario':<30} {'Evidence':<35} ", end="")
        for value in self.domains[query_var]:
            print(f"{value:<12}", end="")
        print()
        print("-" * 100)
        
        for name, evidence, result in results:
            evidence_str = ', '.join([f"{k}={v}" for k, v in evidence.items()])
            print(f"{name:<30} {evidence_str:<35} ", end="")
            for value in self.domains[query_var]:
                print(f"{result[value]:.4f} ({result[value]*100:.1f}%)  ", end="")
            print()
        
        # Print analysis
        print(f"\n{'='*80}")
        print("Amalysis of Differences")
        print(f"{'='*80}\n")
        
        for i, (name1, _, result1) in enumerate(results):
            for j, (name2, _, result2) in enumerate(results[i+1:], i+1):
                print(f"\nComparing '{name1}' vs '{name2}':")
                for value in self.domains[query_var]:
                    diff = result2[value] - result1[value]
                    print(f"  P({query_var}={value}): {result1[value]:.3f} → {result2[value]:.3f} (change: {diff:+.3f})")


class InteractiveBayesianNetwork:
    """Interactive interface for Bayesian network"""
    
    def __init__(self):
        self.bn = BayesianNetwork()
    
    def run(self):
        """Run interactive mode"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "INTERACTIVE BAYESIAN NETWORK".center(78) + "║")
        print("║" + "Crowding Risk Prediction System".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        
        while True:
            print("\n\nOptions:")
            print("  1. Query crowding risk (custom evidence)")
            print("  2. Run predefined scenarios")
            print("  3. Compare multiple scenarios")
            print("  4. View network structure")
            print("  5. Show CPT for a variable")
            print("  6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.custom_query()
            elif choice == '2':
                self.predefined_scenarios()
            elif choice == '3':
                self.compare_multiple_scenarios()
            elif choice == '4':
                self.bn.print_network_structure()
            elif choice == '5':
                self.show_cpt()
            elif choice == '6':
                print("\nThank you for using the Bayesian Network system!")
                break
            else:
                print("\nInvalid option. Please try again.")
    
    def custom_query(self):
        """Allow user to specify custom evidence"""
        print("\n" + "="*80)
        print("Custom Query")
        print("="*80 + "\n")
        
        evidence = {}
        
        # Get Mode
        mode = self._get_input("Mode", ['today', 'future'])
        if mode:
            evidence['Mode'] = mode
        
        # Get Time_Of_Day
        time = self._get_input("Time_Of_Day", ['peak', 'off_peak'])
        if time:
            evidence['Time_Of_Day'] = time
        
        # Get Station_Type
        station = self._get_input("Station_Type", ['interchange', 'regular', 'airport'])
        if station:
            evidence['Station_Type'] = station
        
        if not evidence:
            print("\nNo evidence provided. Cannot perform inference.")
            return
        
        # Query
        result, steps = self.bn.infer(evidence, 'Crowding_Risk', show_steps=True)
        
        # Show recommendation
        print(f"\n{'='*80}")
        print("Reccommendations")
        print(f"{'='*80}\n")
        
        max_risk = max(result.items(), key=lambda x: x[1])
        if max_risk[0] == 'high' and max_risk[1] > 0.5:
            print("High Crowding Risk")
            print("   Recommendations:")
            print("   • Consider alternative routes")
            print("   • Allow extra travel time")
            print("   • Avoid peak hours if possible")
        elif max_risk[0] == 'low' and max_risk[1] > 0.5:
            print("Low Crowding Risk")
            print("   Good time to travel!")
        else:
            print("Moderate Crowding Risk")
            print("   Normal travel conditions expected")
    
    def predefined_scenarios(self):
        """Run predefined test scenarios"""
        print("\n" + "="*80)
        print("Predefined Scenarios")
        print("="*80 + "\n")
        
        scenarios = [
            ("Today Peak Interchange", {'Mode': 'today', 'Time_Of_Day': 'peak', 'Station_Type': 'interchange'}),
            ("Future Peak Interchange", {'Mode': 'future', 'Time_Of_Day': 'peak', 'Station_Type': 'interchange'}),
            ("Today Off-Peak Airport", {'Mode': 'today', 'Time_Of_Day': 'off_peak', 'Station_Type': 'airport'}),
            ("Future Off-Peak Airport", {'Mode': 'future', 'Time_Of_Day': 'off_peak', 'Station_Type': 'airport'}),
        ]
        
        print("Available scenarios:")
        for i, (name, evidence) in enumerate(scenarios, 1):
            evidence_str = ', '.join([f"{k}={v}" for k, v in evidence.items()])
            print(f"  {i}. {name}: {evidence_str}")
        
        choice = input("\nSelect scenario (1-4) or 'a' for all: ").strip()
        
        if choice == 'a':
            for name, evidence in scenarios:
                print(f"\n{'='*80}")
                print(f"Scenario: {name}")
                print(f"{'='*80}")
                result, _ = self.bn.infer(evidence, 'Crowding_Risk', show_steps=True)
                input("\nPress Enter to continue...")
        elif choice in ['1', '2', '3', '4']:
            name, evidence = scenarios[int(choice) - 1]
            print(f"\n{'='*80}")
            print(f"Scenario: {name}")
            print(f"{'='*80}")
            result, _ = self.bn.infer(evidence, 'Crowding_Risk', show_steps=True)
    
    def compare_multiple_scenarios(self):
        """Compare predefined scenarios"""
        print("\n" + "="*80)
        print("Scenario Comparison")
        print("="*80 + "\n")
        
        scenarios = [
            ("Today Peak Interchange", {'Mode': 'today', 'Time_Of_Day': 'peak', 'Station_Type': 'interchange'}),
            ("Future Peak Interchange", {'Mode': 'future', 'Time_Of_Day': 'peak', 'Station_Type': 'interchange'}),
            ("Today Off-Peak Regular", {'Mode': 'today', 'Time_Of_Day': 'off_peak', 'Station_Type': 'regular'}),
            ("Future Off-Peak Regular", {'Mode': 'future', 'Time_Of_Day': 'off_peak', 'Station_Type': 'regular'}),
        ]
        
        self.bn.compare_scenarios(scenarios, 'Crowding_Risk')
    
    def show_cpt(self):
        """Show CPT for a specific variable"""
        print("\n" + "="*80)
        print("Conditional Probability Tables")
        print("="*80 + "\n")
        
        print("Available variables:")
        for i, var in enumerate(sorted(self.bn.variables), 1):
            print(f"  {i}. {var}")
        
        choice = input("\nSelect variable (1-7): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 7:
            var = sorted(self.bn.variables)[int(choice) - 1]
            self.bn.cpts[var].print_table()
    
    def _get_input(self, variable: str, options: List[str]) -> Optional[str]:
        """Get input for a variable"""
        print(f"\n{variable} options: {', '.join(options)}")
        value = input(f"Enter {variable} (or press Enter to skip): ").strip().lower()
        
        if not value:
            return None
        
        if value in options:
            return value
        
        # Try partial match
        matches = [opt for opt in options if opt.startswith(value)]
        if len(matches) == 1:
            print(f"  Interpreting as: {matches[0]}")
            return matches[0]
        
        print(f"  Invalid value. Skipping {variable}.")
        return None


if __name__ == "__main__":
    app = InteractiveBayesianNetwork()
    app.run()
