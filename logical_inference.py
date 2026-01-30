"""
Logical Inference
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Clause:
    """Represents a clause in CNF (Conjunctive Normal Form)"""
    literals: Set[str]
    
    def __repr__(self):
        if not self.literals:
            return "⊥"  # Empty clause (contradiction)
        return " ∨ ".join(sorted(self.literals))
    
    def __hash__(self):
        return hash(frozenset(self.literals))
    
    def __eq__(self, other):
        return self.literals == other.literals


class ResolutionStep:
    """single resolution step"""
    def __init__(self, step_num: int, clause1: Clause, clause2: Clause, 
                 resolvent: Optional[Clause], complementary_literal: str = None):
        self.step_num = step_num
        self.clause1 = clause1
        self.clause2 = clause2
        self.resolvent = resolvent
        self.complementary_literal = complementary_literal
    
    def __repr__(self):
        if self.resolvent is None:
            return f"Step {self.step_num}: No resolution possible"
        return f"Step {self.step_num}: {self.clause1} ⊗ {self.clause2} → {self.resolvent}"


class LogicalInferenceEngine:
    """
    Logical inference engine for MRT service rules
    """
    
    def __init__(self):
        self.knowledge_base: Set[Clause] = set()
        self.mode = "today"
        self.resolution_history: List[ResolutionStep] = []
    
    def add_rule(self, clause: Clause):
        """Add a rule to the knowledge base"""
        self.knowledge_base.add(clause)
    
    def set_mode(self, mode: str):
        """Set mode (today or future)"""
        self.mode = mode
    
    def clear_kb(self):
        """Clear knowledge base"""
        self.knowledge_base.clear()
        self.resolution_history.clear()
    
    def _negate_literal(self, literal: str) -> str:
        """Negate a literal"""
        return literal[1:] if literal.startswith("¬") else f"¬{literal}"
    
    def _resolve(self, clause1: Clause, clause2: Clause) -> Tuple[Optional[Clause], Optional[str]]:
        """
        Apply resolution rule to two clauses
        
        Returns:
            (resolvent clause, complementary literal used) or (None, None)
        """
        for lit1 in clause1.literals:
            neg_lit1 = self._negate_literal(lit1)
            if neg_lit1 in clause2.literals:
                # Found complementary literals - resolve
                new_literals = (clause1.literals | clause2.literals) - {lit1, neg_lit1}
                return Clause(new_literals), lit1
        return None, None
    
    def is_consistent(self, advisory: Set[Clause], show_steps: bool = False) -> Tuple[bool, Optional[str], List[ResolutionStep]]:
        """
        Check if a set of advisories is internally consistent
        
        Returns:
            (is_consistent, explanation, resolution_steps)
        """
        clauses = self.knowledge_base | advisory
        new_clauses = clauses.copy()
        
        steps = []
        step_num = 0
        max_iterations = 1000
        iteration = 0
        
        if show_steps:
            print("Consistency - Resolution Algorithm")
            print(f"Knowledge Base: {len(self.knowledge_base)} rules")
            print(f"Advisory: {len(advisory)} clauses")
            print(f"Total clauses: {len(clauses)}\n")
            
            print("Initial clauses:")
            for i, clause in enumerate(sorted(clauses, key=lambda c: str(c)), 1):
                print(f"  {i}. {clause}")
            print()
        
        while iteration < max_iterations:
            iteration += 1
            pairs = [(c1, c2) for c1 in new_clauses for c2 in new_clauses if c1 != c2]
            
            if show_steps and iteration == 1:
                print(f"Starting resolution (max {max_iterations} iterations)...\n")
            
            resolvents = set()
            for c1, c2 in pairs:
                resolvent, comp_lit = self._resolve(c1, c2)
                if resolvent is not None:
                    step_num += 1
                    
                    # Check for empty clause (contradiction)
                    if len(resolvent.literals) == 0:
                        steps.append(ResolutionStep(step_num, c1, c2, resolvent, comp_lit))
                        
                        if show_steps:
                            print(f"{'='*80}")
                            print(f"Contradiction found at iteration {iteration}")
                            print(f"{'='*80}\n")
                            print(f"Step {step_num}:")
                            print(f"  Clause 1: {c1}")
                            print(f"  Clause 2: {c2}")
                            print(f"  Complementary literal: {comp_lit} and {self._negate_literal(comp_lit)}")
                            print(f"  Resolvent: ⊥ (empty clause)\n")
                            print("Empty clause derived → Contradiction detected")
                        
                        return False, "Contradiction detected: empty clause derived", steps
                    
                    resolvents.add(resolvent)
                    
                    if show_steps and step_num <= 10:  # Show first 10 resolution steps
                        print(f"Step {step_num} (iteration {iteration}):")
                        print(f"  Resolving: {c1}")
                        print(f"        with: {c2}")
                        print(f"  Complementary: {comp_lit} ⊗ {self._negate_literal(comp_lit)}")
                        print(f"  Resolvent: {resolvent}\n")
                    
                    steps.append(ResolutionStep(step_num, c1, c2, resolvent, comp_lit))
            
            # If no new clauses, stop
            if resolvents.issubset(new_clauses):
                if show_steps:
                    print(f"Resolution complete after {iteration} iterations.")
                    print(f"No new clauses derived after {iteration} iterations.")
                    print(f"Total resolution steps: {step_num}")
                    print(f"Final clause count: {len(new_clauses)}")
                    print(f"\n No contradictions found - Advisory is Consistent")
                break
            
            new_clauses |= resolvents
        
        return True, "No contradictions found", steps
    
    def query(self, query_clause: Clause, show_steps: bool = False) -> Tuple[bool, str, List[ResolutionStep]]:
        """
        Query whether a clause can be derived from knowledge base
        Uses proof by contradiction: add negation of query and look for contradiction
        
        Returns:
            (can_prove, explanation, resolution_steps)
        """
        # Negate the query
        negated_literals = {self._negate_literal(lit) for lit in query_clause.literals}
        negated_query = Clause(negated_literals)
        
        if show_steps:
            print(f"\n{'='*80}")
            print(f"Query: Can we prove {query_clause}?")
            print(f"{'='*80}\n")
            print(f"Method: Proof by contradiction")
            print(f"  Original query: {query_clause}")
            print(f"  Negated query: {negated_query}")
            print(f"\nAdding negated query to KB and checking for contradiction...")
        
        # Add negated query to KB
        test_kb = {negated_query}
        consistent, msg, steps = self.is_consistent(test_kb, show_steps)
        
        if not consistent:
            # Found contradiction → query is proven
            if show_steps:
                print(f"\n{'='*80}")
                print(f"RESULT: Query PROVEN")
                print(f"{'='*80}\n")
                print(f"Contradiction found when assuming ¬({query_clause})")
                print(f"Therefore, {query_clause} must be TRUE")
            return True, f"Query proven true (contradiction found)", steps
        else:
            if show_steps:
                print(f"\n{'='*80}")
                print(f"RESULT: Query CANNOT BE PROVEN")
                print(f"{'='*80}\n")
                print(f"No contradiction found when assuming ¬({query_clause})")
                print(f"Therefore, {query_clause} cannot be proven from KB")
            return False, "Query cannot be proven from knowledge base", steps
    
    def is_route_valid(self, route: List[str], advisory: Set[Clause], 
                       show_steps: bool = False) -> Tuple[bool, str]:
        """
        Check if a proposed route is valid under given advisories
        
        Args:
            route: List of station codes
            advisory: Set of advisory clauses
        
        Returns:
            (is_valid, explanation)
        """
        if show_steps:
            print(f"\n{'='*80}")
            print(f"ROUTE VALIDATION")
            print(f"{'='*80}\n")
            print(f"Route: {' → '.join(route)}")
            print(f"Advisory clauses: {len(advisory)}\n")
        
        # First check advisory consistency
        consistent, msg, _ = self.is_consistent(advisory, show_steps=False)
        if not consistent:
            return False, f"Advisory set is inconsistent: {msg}"
        
        # Create facts about the route
        route_facts = set()
        for i in range(len(route) - 1):
            station1, station2 = route[i], route[i+1]
            route_facts.add(Clause({f"uses_{station1}_{station2}"}))
        
        if show_steps:
            print("Route facts:")
            for fact in route_facts:
                print(f"  {fact}")
            print()
        
        # Check if route facts contradict advisory
        combined = self.knowledge_base | advisory | route_facts
        
        # Try to derive contradiction
        new_clauses = combined.copy()
        max_iterations = 500
        
        for iteration in range(max_iterations):
            pairs = [(c1, c2) for c1 in new_clauses for c2 in new_clauses if c1 != c2]
            
            for c1, c2 in pairs:
                resolvent, _ = self._resolve(c1, c2)
                if resolvent and len(resolvent.literals) == 0:
                    if show_steps:
                        print(f"CONTRADICTION FOUND")
                        print(f"  Route violates advisory constraints")
                    return False, "Route violates advisory constraints"
            
            # Add new resolvents
            resolvents = set()
            for c1, c2 in pairs:
                resolvent, _ = self._resolve(c1, c2)
                if resolvent:
                    resolvents.add(resolvent)
            
            if resolvents.issubset(new_clauses):
                break
            
            new_clauses |= resolvents
        
        if show_steps:
            print(f"NO CONTRADICTION")
            print(f"  Route is valid under given advisories")
        
        return True, "Route is valid under given advisories"
    
    def print_knowledge_base(self):
        """Print current knowledge base"""
        print(f"\n{'='*80}")
        print(f"KNOWLEDGE BASE ({self.mode.upper()} MODE)")
        print(f"{'='*80}\n")
        print(f"Total rules: {len(self.knowledge_base)}\n")
        
        for i, clause in enumerate(sorted(self.knowledge_base, key=lambda c: str(c)), 1):
            print(f"  {i}. {clause}")


def initialize_mrt_rules(mode: str = "today") -> LogicalInferenceEngine:
    """
    Initialize logical inference engine with MRT operational rules
    
    Rules cover:
    1. TELe/CRL conversion rules (Future Mode)
    2. Disruption handling
    3. Transfer requirements
    4. Service adjustment constraints
    5. Peak hour operations
    """
    engine = LogicalInferenceEngine()
    engine.set_mode(mode)
    
    # Rule 1: Systems integration → service adjustment required
    engine.add_rule(Clause({"¬systems_integration", "service_adjustment_required"}))
    
    # Rule 2: Tanah Merah conversion → require alternative path
    engine.add_rule(Clause({"¬tanah_merah_conversion", "require_alternative_path"}))
    
    # Rule 3: TELe operational → Changi accessible via TEL (Future mode)
    if mode == "future":
        engine.add_rule(Clause({"¬tele_operational", "changi_via_tel_available"}))
        engine.add_rule(Clause({"tele_operational"}))  # Fact
    
    # Rule 4: T5 accessible → both TEL and CRL available
    if mode == "future":
        engine.add_rule(Clause({"¬t5_accessible", "tel_at_t5"}))
        engine.add_rule(Clause({"¬t5_accessible", "crl_at_t5"}))
        engine.add_rule(Clause({"t5_accessible"}))  # Fact
    
    # Rule 5: Peak hours → crowding penalty
    engine.add_rule(Clause({"¬peak_hours", "crowding_penalty_active"}))
    
    # Rule 6: Interchange transfer → minimum transfer time
    engine.add_rule(Clause({"¬interchange_transfer", "min_transfer_time_3min"}))
    
    # Rule 7: Conversion complete → stations become TEL
    if mode == "future":
        engine.add_rule(Clause({"¬conversion_complete", "changi_airport_is_tel"}))
        engine.add_rule(Clause({"¬conversion_complete", "expo_is_tel"}))
        engine.add_rule(Clause({"¬conversion_complete", "tanah_merah_is_tel_ewl_interchange"}))
        engine.add_rule(Clause({"conversion_complete"}))  # Fact
    
    # Rule 8: Reduced frequency → increased wait time
    engine.add_rule(Clause({"¬reduced_frequency", "increased_wait_time"}))
    
    # Rule 9: Direct route available → can use direct
    engine.add_rule(Clause({"¬direct_route_available", "can_use_direct"}))
    
    # Rule 10: Many transfers → suboptimal route
    engine.add_rule(Clause({"¬many_transfers", "suboptimal_route"}))
    
    # Rule 11: Airport station → luggage space required
    engine.add_rule(Clause({"¬airport_station", "luggage_space_required"}))
    
    return engine


class InteractiveLogicalInference:
    """Interactive interface for logical inference"""
    
    def __init__(self):
        self.engine_today = initialize_mrt_rules(mode="today")
        self.engine_future = initialize_mrt_rules(mode="future")
        self.current_engine = self.engine_today
        self.current_mode = "today"
    
    def run(self):
        """Run interactive mode"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "INTERACTIVE LOGICAL INFERENCE SYSTEM".center(78) + "║")
        print("║" + "MRT Service Rules & Advisory Validation".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        
        while True:
            print(f"\n\nCurrent Mode: {self.current_mode.upper()}")
            print("\nOptions:")
            print("  1. Check advisory consistency")
            print("  2. Query a proposition")
            print("  3. Validate a route")
            print("  4. View knowledge base")
            print("  5. Test predefined scenarios")
            print("  6. Switch mode (Today/Future)")
            print("  7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.check_consistency()
            elif choice == '2':
                self.query_proposition()
            elif choice == '3':
                self.validate_route()
            elif choice == '4':
                self.current_engine.print_knowledge_base()
            elif choice == '5':
                self.test_scenarios()
            elif choice == '6':
                self.switch_mode()
            elif choice == '7':
                print("\nThank you for using the Logical Inference System!")
                break
            else:
                print("\nInvalid option. Please try again.")
    
    def check_consistency(self):
        """Check if an advisory is consistent"""
        print("\n" + "="*80)
        print("Check Advisory Consistency")
        print("="*80 + "\n")
        
        print("Common advisories:")
        print("  1. Segment EW5-CG2 disrupted")
        print("  2. Systems integration work")
        print("  3. Contradictory advisories (test)")
        print("  4. Custom advisory")
        
        choice = input("\nSelect advisory (1-4): ").strip()
        
        if choice == '1':
            advisory = {Clause({"¬service_EW5_CG2"})}
            print("\nAdvisory: Expo-Changi Airport segment suspended")
        elif choice == '2':
            advisory = {
                Clause({"systems_integration"}),
                Clause({"¬service_EW4_EW5"})
            }
            print("\nAdvisory: Systems integration work affecting Tanah Merah-Expo")
        elif choice == '3':
            advisory = {
                Clause({"service_available"}),
                Clause({"¬service_available"})
            }
            print("\nAdvisory: Service both available and not available (contradiction test)")
        elif choice == '4':
            print("\nEnter clauses (format: 'literal1 v literal2 v ...')")
            print("Use '¬' for negation. Type 'done' when finished.")
            advisory = set()
            while True:
                clause_str = input("Clause: ").strip()
                if clause_str.lower() == 'done':
                    break
                literals = {lit.strip() for lit in clause_str.split('v')}
                advisory.add(Clause(literals))
            
            if not advisory:
                print("\nNo advisory clauses provided.")
                return
        else:
            print("\nInvalid choice.")
            return
        
        # Check consistency
        consistent, msg, steps = self.current_engine.is_consistent(advisory, show_steps=True)
        
        input("\nPress Enter to continue...")
    
    def query_proposition(self):
        """Query whether a proposition can be proven"""
        print("\n" + "="*80)
        print("QUERY PROPOSITION")
        print("="*80 + "\n")
        
        if self.current_mode == "future":
            print("Example queries for FUTURE mode:")
            print("  1. changi_via_tel_available")
            print("  2. tel_at_t5")
            print("  3. crl_at_t5")
            print("  4. changi_airport_is_tel")
        else:
            print("Example queries for TODAY mode:")
            print("  1. service_adjustment_required (if systems_integration)")
            print("  2. crowding_penalty_active (if peak_hours)")
        
        print("\nEnter proposition to query:")
        prop = input("> ").strip()
        
        if not prop:
            print("\nNo proposition provided.")
            return
        
        query = Clause({prop})
        can_prove, msg, steps = self.current_engine.query(query, show_steps=True)
        
        input("\nPress Enter to continue...")
    
    def validate_route(self):
        """Validate a route against advisories"""
        print("\n" + "="*80)
        print("Route Validation")
        print("="*80 + "\n")
        
        print("Enter route as space-separated station codes:")
        print("Example: EW5 CG2")
        route_str = input("> ").strip()
        
        if not route_str:
            print("\nNo route provided.")
            return
        
        route = route_str.split()
        
        print("\nEnter advisory (or press Enter for no disruptions):")
        print("Example: ¬service_EW5_CG2")
        advisory_str = input("> ").strip()
        
        advisory = set()
        if advisory_str:
            advisory.add(Clause({advisory_str}))
        
        valid, msg = self.current_engine.is_route_valid(route, advisory, show_steps=True)
        
        print(f"\n{'='*80}")
        print(f"Result: {'Valid' if valid else 'Invalid'}")
        print(f"{'='*80}")
        print(f"\n{msg}")
        
        input("\nPress Enter to continue...")
    
    def test_scenarios(self):
        """Test predefined scenarios"""
        print("\n" + "="*80)
        print("Predefined Test Cases")
        print("="*80 + "\n")
        
        scenarios = [
            ("Normal Operations", set(), True),
            ("Segment Disruption", {Clause({"¬service_EW5_CG2"})}, True),
            ("Contradictory Advisory", {
                Clause({"service_EW4_EW6"}),
                Clause({"¬service_EW4_EW6"})
            }, False),
        ]
        
        if self.current_mode == "future":
            scenarios.append((
                "TELe Availability Query",
                set(),
                "query:changi_via_tel_available"
            ))
        
        for i, scenario_data in enumerate(scenarios, 1):
            if len(scenario_data) == 3 and isinstance(scenario_data[2], str) and scenario_data[2].startswith("query:"):
                name, _, query_str = scenario_data
                print(f"\n{'='*80}")
                print(f"Scenario {i}: {name}")
                print(f"{'='*80}")
                
                prop = query_str.split(":")[1]
                query = Clause({prop})
                can_prove, msg, _ = self.current_engine.query(query, show_steps=True)
                
            else:
                name, advisory, expected = scenario_data
                print(f"\n{'='*80}")
                print(f"Scenario {i}: {name}")
                print(f"{'='*80}")
                
                consistent, msg, _ = self.current_engine.is_consistent(advisory, show_steps=True)
                
                if consistent == expected:
                    print(f"\Result matches expected: {expected}")
                else:
                    print(f"\nUnexpected result: got {consistent}, expected {expected}")
            
            input("\nPress Enter for next scenario...")
    
    def switch_mode(self):
        """Switch between Today and Future modes"""
        if self.current_mode == "today":
            self.current_mode = "future"
            self.current_engine = self.engine_future
        else:
            self.current_mode = "today"
            self.current_engine = self.engine_today
        
        print(f"\nSwitched to {self.current_mode.upper()} mode")
        print(f"  Knowledge base: {len(self.current_engine.knowledge_base)} rules")


if __name__ == "__main__":
    app = InteractiveLogicalInference()
    app.run()
