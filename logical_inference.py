"""
Logical Inference for MRT Service Rules & Advisory Consistency
Uses propositional logic and resolution-based inference
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Clause:
    """Represents a clause in CNF (Conjunctive Normal Form)"""
    literals: Set[str]  # Set of literals (positive or negative)
    
    def __repr__(self):
        if not self.literals:
            return "⊥"  # Empty clause (contradiction)
        return " ∨ ".join(sorted(self.literals))
    
    def __hash__(self):
        return hash(frozenset(self.literals))
    
    def __eq__(self, other):
        return self.literals == other.literals


class LogicalInferenceEngine:
    """
    Logical inference engine for MRT service rules
    
    Uses resolution-based inference to:
    1. Validate routes against operational rules
    2. Detect contradictions in service advisories
    3. Deduce whether proposed routes are valid
    """
    
    def __init__(self):
        self.knowledge_base: Set[Clause] = set()
        self.mode = "today"  # or "future"
    
    def add_rule(self, clause: Clause):
        """Add a rule to the knowledge base"""
        self.knowledge_base.add(clause)
    
    def set_mode(self, mode: str):
        """Set network mode (today or future)"""
        self.mode = mode
    
    def _negate_literal(self, literal: str) -> str:
        """Negate a literal"""
        return literal[1:] if literal.startswith("¬") else f"¬{literal}"
    
    def _resolve(self, clause1: Clause, clause2: Clause) -> Optional[Clause]:
        """
        Apply resolution rule to two clauses
        
        If one clause contains P and another contains ¬P,
        create a resolvent with all other literals
        """
        for lit1 in clause1.literals:
            neg_lit1 = self._negate_literal(lit1)
            if neg_lit1 in clause2.literals:
                # Found complementary literals - resolve
                new_literals = (clause1.literals | clause2.literals) - {lit1, neg_lit1}
                return Clause(new_literals)
        return None
    
    def is_consistent(self, advisory: Set[Clause]) -> Tuple[bool, Optional[str]]:
        """
        Check if a set of advisories is internally consistent
        
        Returns:
            (is_consistent, explanation)
        """
        # Combine knowledge base with advisory
        clauses = self.knowledge_base | advisory
        new_clauses = clauses.copy()
        
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            pairs = [(c1, c2) for c1 in new_clauses for c2 in new_clauses if c1 != c2]
            
            resolvents = set()
            for c1, c2 in pairs:
                resolvent = self._resolve(c1, c2)
                if resolvent is not None:
                    # Empty clause means contradiction
                    if len(resolvent.literals) == 0:
                        return False, "Contradiction detected: empty clause derived"
                    resolvents.add(resolvent)
            
            # If no new clauses, we're done
            if resolvents.issubset(new_clauses):
                break
            
            new_clauses |= resolvents
        
        return True, "No contradictions found"
    
    def is_route_valid(self, route: List[str], advisory: Set[Clause]) -> Tuple[bool, str]:
        """
        Check if a proposed route is valid under given advisories
        
        Args:
            route: List of station codes
            advisory: Set of advisory clauses
        
        Returns:
            (is_valid, explanation)
        """
        # First check advisory consistency
        consistent, msg = self.is_consistent(advisory)
        if not consistent:
            return False, f"Advisory set is inconsistent: {msg}"
        
        # Create facts about the route
        route_facts = set()
        for i in range(len(route) - 1):
            station1, station2 = route[i], route[i+1]
            route_facts.add(Clause({f"uses_{station1}_{station2}"}))
        
        # Check if route facts contradict advisory
        combined = self.knowledge_base | advisory | route_facts
        
        # Try to derive contradiction
        new_clauses = combined.copy()
        max_iterations = 500
        
        for _ in range(max_iterations):
            pairs = [(c1, c2) for c1 in new_clauses for c2 in new_clauses if c1 != c2]
            
            for c1, c2 in pairs:
                resolvent = self._resolve(c1, c2)
                if resolvent and len(resolvent.literals) == 0:
                    return False, "Route violates advisory constraints"
            
            # Add new resolvents
            resolvents = {self._resolve(c1, c2) for c1, c2 in pairs if self._resolve(c1, c2)}
            resolvents.discard(None)
            
            if resolvents.issubset(new_clauses):
                break
            
            new_clauses |= resolvents
        
        return True, "Route is valid under given advisories"
    
    def query(self, query_clause: Clause) -> Tuple[bool, str]:
        """
        Query whether a clause can be derived from knowledge base
        
        Uses proof by contradiction: add negation of query and look for contradiction
        """
        # Negate the query
        negated_query = Clause({self._negate_literal(lit) for lit in query_clause.literals})
        
        # Add negated query to KB
        clauses = self.knowledge_base | {negated_query}
        new_clauses = clauses.copy()
        
        max_iterations = 1000
        for iteration in range(max_iterations):
            pairs = [(c1, c2) for c1 in new_clauses for c2 in new_clauses if c1 != c2]
            
            resolvents = set()
            for c1, c2 in pairs:
                resolvent = self._resolve(c1, c2)
                if resolvent is not None:
                    if len(resolvent.literals) == 0:
                        return True, f"Query proven true (contradiction found at iteration {iteration})"
                    resolvents.add(resolvent)
            
            if resolvents.issubset(new_clauses):
                break
            
            new_clauses |= resolvents
        
        return False, "Query cannot be proven from knowledge base"


def initialize_mrt_rules(mode: str = "today") -> LogicalInferenceEngine:
    """
    Initialize logical inference engine with MRT operational rules
    
    Rules cover:
    1. TELe/CRL conversion rules (Future Mode)
    2. Disruption handling
    3. Transfer requirements
    4. Service adjustment constraints
    5. Peak hour operations
    6. Safety requirements
    7. Interchange restrictions
    8. Line-specific operations
    9. System integration constraints
    10. Emergency protocols
    """
    engine = LogicalInferenceEngine()
    engine.set_mode(mode)
    
    # Rule 1: If station is closed, cannot use routes through it
    # ¬closed_X ∨ ¬uses_path_through_X
    # (If X is closed, then we cannot use a path through X)
    
    # Rule 2: If segment has disruption, must use alternative route
    # disruption_X_Y → must_avoid_X_Y
    # Converted to CNF: ¬disruption_X_Y ∨ must_avoid_X_Y
    
    # Rule 3: During systems integration, EWL Changi branch has service adjustments
    # systems_integration → service_adjustment_EWL_changi
    # CNF: ¬systems_integration ∨ service_adjustment_EWL_changi
    engine.add_rule(Clause({"¬systems_integration", "service_adjustment_required"}))
    
    # Rule 4: If Tanah Merah is undergoing conversion, alternative paths required
    # tanah_merah_conversion → require_alternative_path
    # CNF: ¬tanah_merah_conversion ∨ require_alternative_path
    engine.add_rule(Clause({"¬tanah_merah_conversion", "require_alternative_path"}))
    
    # Rule 5: TELe extension operational → Changi Airport accessible via TEL
    # tele_operational → changi_via_tel_available
    # CNF: ¬tele_operational ∨ changi_via_tel_available
    if mode == "future":
        engine.add_rule(Clause({"¬tele_operational", "changi_via_tel_available"}))
        engine.add_rule(Clause({"tele_operational"}))  # Fact: TELe is operational in future mode
    
    # Rule 6: If T5 is accessible, it's an interchange with both TEL and CRL
    # t5_accessible → (tel_at_t5 ∧ crl_at_t5)
    # CNF: ¬t5_accessible ∨ tel_at_t5
    #      ¬t5_accessible ∨ crl_at_t5
    if mode == "future":
        engine.add_rule(Clause({"¬t5_accessible", "tel_at_t5"}))
        engine.add_rule(Clause({"¬t5_accessible", "crl_at_t5"}))
        engine.add_rule(Clause({"t5_accessible"}))  # Fact: T5 is accessible in future mode
    
    # Rule 7: Peak hours → higher crowding penalty at major interchanges
    # peak_hours → crowding_penalty_interchanges
    # CNF: ¬peak_hours ∨ crowding_penalty_interchanges
    engine.add_rule(Clause({"¬peak_hours", "crowding_penalty_active"}))
    
    # Rule 8: Interchange transfer → minimum 3 minute transfer time
    # interchange_transfer → min_3min_transfer
    # CNF: ¬interchange_transfer ∨ min_3min_transfer
    engine.add_rule(Clause({"¬interchange_transfer", "min_transfer_time_3min"}))
    
    # Rule 9: If segment is under maintenance, no service on that segment
    # maintenance_X_Y → ¬service_X_Y
    # CNF: ¬maintenance_X_Y ∨ ¬service_X_Y
    
    # Rule 10: During conversion, EWL Changi stations become TEL (future)
    # conversion_complete → (changi_airport_is_tel ∧ expo_is_tel ∧ tanah_merah_is_interchange)
    if mode == "future":
        engine.add_rule(Clause({"¬conversion_complete", "changi_airport_is_tel"}))
        engine.add_rule(Clause({"¬conversion_complete", "expo_is_tel"}))
        engine.add_rule(Clause({"¬conversion_complete", "tanah_merah_is_tel_ewl_interchange"}))
        engine.add_rule(Clause({"conversion_complete"}))  # Fact: Conversion complete in future
    
    # Rule 11: Emergency situation → all affected stations closed
    # emergency_X → closed_X
    # CNF: ¬emergency_X ∨ closed_X
    
    # Rule 12: Reduced frequency segment → longer wait times
    # reduced_frequency_X_Y → increased_wait_time_X_Y
    # CNF: ¬reduced_frequency_X_Y ∨ increased_wait_time_X_Y
    engine.add_rule(Clause({"¬reduced_frequency", "increased_wait_time"}))
    
    # Rule 13: Direct route available → prefer over transfer route (optimization hint)
    # direct_available ∧ ¬disruption → prefer_direct
    # This is more complex, but simplified: direct_available → can_use_direct
    # CNF: ¬direct_available ∨ can_use_direct
    engine.add_rule(Clause({"¬direct_route_available", "can_use_direct"}))
    
    # Rule 14: Maximum 2 transfers recommended for optimal journey
    # transfers > 2 → suboptimal_route
    # Represented as: many_transfers → suboptimal
    # CNF: ¬many_transfers ∨ suboptimal_route
    engine.add_rule(Clause({"¬many_transfers", "suboptimal_route"}))
    
    # Rule 15: Airport stations have luggage space requirements
    # airport_station → luggage_space_required
    # CNF: ¬airport_station ∨ luggage_space_required
    engine.add_rule(Clause({"¬airport_station", "luggage_space_required"}))
    
    return engine


def test_logical_inference():
    """Test the logical inference system with MRT scenarios"""
    
    print("=== Logical Inference System Test ===\n")
    
    # Test 1: Today Mode - Normal operations
    print("Test 1: Today Mode - Check if advisory set is consistent")
    engine_today = initialize_mrt_rules(mode="today")
    
    # Advisory: Systems integration work at Expo
    advisory1 = {
        Clause({"systems_integration"}),
        Clause({"¬service_EW5_CG2"}),  # No service Expo to Changi Airport
    }
    
    consistent, msg = engine_today.is_consistent(advisory1)
    print(f"Advisory 1 consistent: {consistent}")
    print(f"Explanation: {msg}\n")
    
    # Test 2: Future Mode - TELe operational
    print("Test 2: Future Mode - Verify TELe rules")
    engine_future = initialize_mrt_rules(mode="future")
    
    # Query: Can we reach Changi Airport via TEL?
    query = Clause({"changi_via_tel_available"})
    result, explanation = engine_future.query(query)
    print(f"Query 'Changi accessible via TEL': {result}")
    print(f"Explanation: {explanation}\n")
    
    # Test 3: Contradictory advisory
    print("Test 3: Detect contradiction in advisory")
    advisory_contradiction = {
        Clause({"service_EW4_EW5"}),      # Service available Tanah Merah to Expo
        Clause({"¬service_EW4_EW5"}),    # No service Tanah Merah to Expo
    }
    
    consistent, msg = engine_today.is_consistent(advisory_contradiction)
    print(f"Contradictory advisory consistent: {consistent}")
    print(f"Explanation: {msg}\n")
    
    # Test 4: Route validation
    print("Test 4: Validate route under service disruption")
    route = ["EW5", "CG2"]  # Expo to Changi Airport
    advisory_disruption = {
        Clause({"¬service_EW5_CG2"}),  # Segment disrupted
    }
    
    valid, explanation = engine_today.is_route_valid(route, advisory_disruption)
    print(f"Route {' -> '.join(route)} valid: {valid}")
    print(f"Explanation: {explanation}\n")
    
    # Test 5: Complex scenario - Future mode with T5
    print("Test 5: Future Mode - T5 interchange capabilities")
    query_t5 = Clause({"tel_at_t5", "crl_at_t5"})
    
    # We need both literals to be true, so we query each separately
    result_tel, _ = engine_future.query(Clause({"tel_at_t5"}))
    result_crl, _ = engine_future.query(Clause({"crl_at_t5"}))
    
    print(f"TEL at T5: {result_tel}")
    print(f"CRL at T5: {result_crl}")
    print(f"Both available: {result_tel and result_crl}\n")


if __name__ == "__main__":
    test_logical_inference()
