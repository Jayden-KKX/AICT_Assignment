
import pandas as pd
import math
import random
import heapq
from collections import defaultdict
import os

# =========================================================
# 1) Load MRT data from data folder (nodes.csv, edges.csv)
# =========================================================

DATA_PATH = "data"  

def load_mrt_from_zip(data_path: str):
    nodes = pd.read_csv(os.path.join(data_path, "nodes.csv"))
    edges = pd.read_csv(os.path.join(data_path, "edges.csv"))

    # Station -> interchange? transfer penalty
    station_info = {}
    for _, row in nodes.iterrows():
        station_info[row["Station_Name"]] = {
            "is_interchange": str(row["Is_Interchange"]).strip().lower() == "yes",
            "transfer_penalty": float(row["Transfer_Penalty_Min"]),
        }

    # Build adjacency list from edges
    # Each directed edge: u -> v with (line, weight)
    adj = defaultdict(list)
    for _, row in edges.iterrows():
        u = row["From_Station"]
        v = row["To_Station"]
        line = row["Line"]
        w = float(row["Total_Edge_Weight_Min"])
        bidir = str(row["Bidirectional"]).strip().lower() == "yes"

        adj[u].append((v, line, w))
        if bidir:
            adj[v].append((u, line, w))

    return adj, station_info


# =========================================================
# 2) Shortest path with transfer penalties (line-aware)
#    State = (station, current_line)
# =========================================================

def shortest_path_line_aware(
    adj,
    station_info,
    start: str,
    goal: str,
    closed_segments=None,
    blocked_transfers=None,
    edge_weight_overrides=None,
):
    """
    Dijkstra on expanded state space (station, current_line),
    adding transfer penalty when line changes at an interchange station.

    closed_segments: set of (u, v, line) segments that are forbidden (both directions handled)
    blocked_transfers: set of interchange stations where transfers are forbidden
    edge_weight_overrides: dict {(u,v,line): new_weight} used to bias alternate paths
    """

    closed_segments = closed_segments or set()
    blocked_transfers = blocked_transfers or set()
    edge_weight_overrides = edge_weight_overrides or {}

    # Helper: check if segment is closed (in either direction)
    def is_closed(u, v, line):
        return (u, v, line) in closed_segments or (v, u, line) in closed_segments

    # Priority queue: (cost, station, current_line)
    pq = []
    heapq.heappush(pq, (0.0, start, None))

    # dist and parent pointers
    dist = {}
    parent = {}  # key: (station, line) -> prev (station, line) and chosen edge info
    dist[(start, None)] = 0.0

    while pq:
        cost, u, cur_line = heapq.heappop(pq)

        if (u, cur_line) in dist and cost > dist[(u, cur_line)]:
            continue

        # Stop when reaching goal (first time popped is optimal for that state)
        if u == goal:
            # Reconstruct path of stations + lines
            return reconstruct_path(parent, (u, cur_line))

        for v, next_line, base_w in adj.get(u, []):
            if is_closed(u, v, next_line):
                continue

            # If transfer is blocked at u, disallow line changes there
            if u in blocked_transfers and cur_line is not None and next_line != cur_line:
                continue

            # Edge weight override (used for neighbour generation)
            w = edge_weight_overrides.get((u, v, next_line), base_w)

            # Transfer penalty applies when line changes at interchange station u
            transfer_pen = 0.0
            if cur_line is not None and next_line != cur_line:
                if station_info.get(u, {}).get("is_interchange", False):
                    transfer_pen = station_info[u]["transfer_penalty"]
                else:
                    # If not an interchange, line change should not happen (you can enforce hard constraint)
                    # Here, we hard-block it to stay realistic.
                    continue

            new_cost = cost + w + transfer_pen
            state = (v, next_line)

            if state not in dist or new_cost < dist[state]:
                dist[state] = new_cost
                parent[state] = ((u, cur_line), (u, v, next_line, w, transfer_pen))
                heapq.heappush(pq, (new_cost, v, next_line))

    return None  # no route found


def reconstruct_path(parent, end_state):
    """Return dict with stations list, lines list (per hop), and total cost."""
    path_edges = []
    cur = end_state
    total_cost = None

    # Find the end state's cost by summing edges during reconstruction
    # (Alternatively store dist; but for simplicity reconstruct sum here)
    while cur in parent:
        prev_state, edge_info = parent[cur]
        path_edges.append(edge_info)  # (u, v, line, w, transfer_pen)
        cur = prev_state

    path_edges.reverse()

    stations = []
    lines = []
    cost = 0.0

    if path_edges:
        stations.append(path_edges[0][0])  # first u
        for (u, v, line, w, tpen) in path_edges:
            stations.append(v)
            lines.append(line)
            cost += (w + tpen)

    return {
        "stations": stations,
        "lines": lines,   # each hop line
        "cost": cost,
        "edges": path_edges,
    }


# =========================================================
# 3) Neighbour generation for SA (route-level)
#    - pick a random subpath in the current route
#    - bias the optimiser to find an alternate subpath by temporarily increasing weights on current subpath edges
# =========================================================

def make_edge_override_for_subpath(path_edges, boost_factor: float):
    """
    Create edge_weight_overrides that discourage using the current subpath.
    boost_factor > 1.0 increases weights to force alternate routes.
    """
    overrides = {}
    for (u, v, line, w, _) in path_edges:
        overrides[(u, v, line)] = w * boost_factor
        overrides[(v, u, line)] = w * boost_factor  # handle reverse
    return overrides


def generate_neighbor_route(
    adj,
    station_info,
    current_route,
    closed_segments,
    blocked_transfers,
    learning_rate,
):
    """
    Generate a neighbour route by re-routing a random segment of the path.
    """
    stations = current_route["stations"]
    if len(stations) < 4:
        return current_route  # too short to meaningfully reroute

    # Choose two cut points i < j
    i = random.randint(0, len(stations) - 3)
    j = random.randint(i + 2, len(stations) - 1)

    a = stations[i]
    b = stations[j]

    # Discourage using the original subpath edges between i..j by boosting their weights
    # Boost strength depends on learning_rate (higher = more willing to deviate)
    original_sub_edges = current_route["edges"][i:j]  # approx aligned; safe enough for demo
    boost_factor = 1.0 + (learning_rate * 5.0)  # e.g., lr=0.1 => 1.5x

    overrides = make_edge_override_for_subpath(original_sub_edges, boost_factor)

    rerouted = shortest_path_line_aware(
        adj,
        station_info,
        start=a,
        goal=b,
        closed_segments=closed_segments,
        blocked_transfers=blocked_transfers,
        edge_weight_overrides=overrides,
    )

    if rerouted is None or len(rerouted["stations"]) < 2:
        return current_route

    # Stitch: prefix + rerouted middle + suffix
    new_stations = stations[:i] + rerouted["stations"] + stations[j+1:]

    # Recompute a clean full route cost by running shortest_path on entire start->goal
    # (This keeps line/transfer accounting correct after stitching)
    full = shortest_path_line_aware(
        adj,
        station_info,
        start=new_stations[0],
        goal=new_stations[-1],
        closed_segments=closed_segments,
        blocked_transfers=blocked_transfers,
        edge_weight_overrides=None,
    )

    return full if full is not None else current_route


# =========================================================
# 4) Simulated Annealing optimiser (Custom SMT)
# =========================================================

def simulated_annealing_route_opt(
    adj,
    station_info,
    start,
    goal,
    params,
    closed_segments=None,
    blocked_transfers=None,
    seed=42,
):
    """
    Custom SMT using Simulated Annealing on the actual MRT graph.
    """
    random.seed(seed)
    closed_segments = closed_segments or set()
    blocked_transfers = blocked_transfers or set()

    # Initial route: best-known by Dijkstra line-aware (baseline)
    current = shortest_path_line_aware(adj, station_info, start, goal,
                                       closed_segments=closed_segments,
                                       blocked_transfers=blocked_transfers)
    if current is None:
        raise ValueError("No route found for baseline under given constraints.")

    best = current

    T = float(params["initial_temperature"])
    cooling = float(params["cooling_rate"])
    iters = int(params["iterations"])
    lr = float(params["learning_rate"])

    history = []

    for k in range(iters):
        neighbor = generate_neighbor_route(
            adj, station_info, current, closed_segments, blocked_transfers, lr
        )
        if neighbor is None:
            continue

        delta = neighbor["cost"] - current["cost"]

        # Accept if better, else accept probabilistically
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current = neighbor

            if current["cost"] < best["cost"]:
                best = current

        history.append((k, T, current["cost"], best["cost"]))
        T *= cooling

    return best, history


# =========================================================
# 5) Preset parameter configurations
# =========================================================

# === Preset parameter configurations ===

PRESETS = {
    "Exploratory": {
        "initial_temperature": 120.0,
        "cooling_rate": 0.90,
        "iterations": 300,
        "learning_rate": 0.20
    },
    "Stable": {
        "initial_temperature": 30.0,
        "cooling_rate": 0.98,
        "iterations": 800,
        "learning_rate": 0.05
    },
    "Custom": {
        "initial_temperature": 80.0,
        "cooling_rate": 0.95,
        "iterations": 600,
        "learning_rate": 0.10
    }
}


# =========================================================
# 6) Example usage (YOU should edit these)
# =========================================================

if __name__ == "__main__":
    adj, station_info = load_mrt_from_zip(DATA_PATH)

    # Choose an OD pair from your actual station names
    start_station = "Changi Airport"
    goal_station = "Outram Park"

    # Disruption constraints (examples)
    # Close a segment:
    closed_segments = {
        ("Expo", "Tanah Merah", "EWL"),  # example closure
    }
    # Block transfers at an interchange:
    blocked_transfers = {
        "City Hall",  # example
    }

    # Run all presets and collect results
    results = []
    
    print("=" * 80)
    print(f"Running Simulated Annealing with Multiple Presets")
    print(f"Route: {start_station} → {goal_station}")
    print("=" * 80)
    print()
    
    for preset_name, params in PRESETS.items():
        print(f"\n{'=' * 80}")
        print(f"Running Preset: {preset_name}")
        print(f"{'=' * 80}")
        print(f"Parameters: {params}")
        print()
        
        best_route, hist = simulated_annealing_route_opt(
            adj, station_info,
            start_station, goal_station,
            params,
            closed_segments=closed_segments,
            blocked_transfers=blocked_transfers,
            seed=7
        )
        
        print(f"\n--- Results for {preset_name} ---")
        print(f"Cost (min): {round(best_route['cost'], 2)}")
        print(f"Number of Stations: {len(best_route['stations'])}")
        print(f"Number of Transfers: {len(set(best_route['lines'])) - 1}")
        print(f"Stations: {' -> '.join(best_route['stations'])}")
        print(f"Lines (per hop): {best_route['lines']}")
        
        # Store results for summary table
        results.append({
            "preset": preset_name,
            "cost": best_route["cost"],
            "stations": len(best_route["stations"]),
            "transfers": len(set(best_route["lines"])) - 1,
            "route": best_route["stations"],
            "lines": best_route["lines"],
            "params": params
        })
    
    # Print summary table
    print("\n\n")
    print("=" * 80)
    print("SUMMARY TABLE - Comparison of All Presets")
    print("=" * 80)
    print()
    print(f"{'Preset':<15} {'Cost (min)':<12} {'Stations':<10} {'Transfers':<10} {'Iterations':<12}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['preset']:<15} {res['cost']:<12.2f} {res['stations']:<10} "
              f"{res['transfers']:<10} {res['params']['iterations']:<12}")
    
    print("-" * 80)
    
    # Find best result
    best_result = min(results, key=lambda x: x["cost"])
    print(f"\n✓ Best Preset: {best_result['preset']} with cost {best_result['cost']:.2f} minutes")
    print("=" * 80)
