"""
Data Loader for MRT Network
"""

import math
import csv
from typing import Dict, List, Tuple, Optional

class MRTStation:
    """MRT station with coordinates and attributes"""
    def __init__(self, code: str, name: str, line: str, lat: float, lon: float, 
                 station_type: str = "Regular", is_interchange: bool = False):
        self.code = code
        self.name = name
        self.line = line
        self.lat = lat
        self.lon = lon
        self.station_type = station_type
        self.is_interchange = is_interchange
    
    def __repr__(self):
        return f"{self.code} {self.name}"
    
    def __eq__(self, other):
        return self.code == other.code if isinstance(other, MRTStation) else False
    
    def __hash__(self):
        return hash(self.code)


class MRTNetwork:
    """
    MRT Network that loads data from CSV files
    Loads both "Today Mode" and "Future Mode" networks
    """
    
    def __init__(self, nodes_file: str, edges_file: str, mode="today"):
        """
        Initialize MRT network from CSV files
        
        Args:
            nodes_file: Path to nodes CSV file
            edges_file: Path to edges CSV file
            mode: "today" for current network, "future" for TELe/CRL network
        """
        self.mode = mode
        self.stations: Dict[str, MRTStation] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}
        self.station_codes: Dict[str, str] = {}  # name -> code mapping
        
        self._load_stations(nodes_file)
        self._load_edges(edges_file)
    
    def _load_stations(self, nodes_file: str):
        """Load stations from CSV file"""
        with open(nodes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['Station_Code']
                name = row['Station_Name']
                line = row['Line']
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                station_type = row['Station_Type']
                is_interchange = row['Is_Interchange'].lower() == 'yes'
                
                # Filter based on mode
                if self.mode == "today" and "CR" in code:
                    continue  # Skip CRL stations in today mode
                
                station = MRTStation(code, name, line, lat, lon, station_type, is_interchange)
                self.stations[code] = station
                self.station_codes[name.lower()] = code
    
    def _load_edges(self, edges_file: str):
        """Load edges from CSV file"""
        with open(edges_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                from_station = row['From_Station']
                to_station = row['To_Station']
                
                # Map station names to codes
                from_code = self._find_station_code(from_station)
                to_code = self._find_station_code(to_station)
                
                if not from_code or not to_code:
                    continue
                
                # Filter based on mode
                if self.mode == "today":
                    # Skip future connections
                    if "Terminal 5" in from_station or "Terminal 5" in to_station:
                        continue
                
                weight = float(row['Total_Edge_Weight_Min'])
                bidirectional = row['Bidirectional'].lower() == 'yes'
                
                self._add_edge(from_code, to_code, weight, bidirectional)
    
    def _find_station_code(self, station_name: str) -> Optional[str]:
        """Find station code by name"""
        # Direct lookup
        code = self.station_codes.get(station_name.lower())
        if code:
            return code
        
        # Try matching with existing station codes
        for code, station in self.stations.items():
            if station.name.lower() == station_name.lower():
                return code
        
        return None
    
    def _add_edge(self, station1: str, station2: str, weight: float, bidirectional: bool = True):
        """Add edge between two stations"""
        if station1 not in self.edges:
            self.edges[station1] = []
        if station2 not in self.edges:
            self.edges[station2] = []
        
        self.edges[station1].append((station2, weight))
        if bidirectional:
            self.edges[station2].append((station1, weight))
    
    def get_neighbors(self, station_code: str) -> List[Tuple[str, float]]:
        """Get neighbors of a station w travel times"""
        return self.edges.get(station_code, [])
    
    def get_station(self, station_code: str) -> Optional[MRTStation]:
        """Get station by code"""
        return self.stations.get(station_code)
    
    def get_station_by_name(self, name: str) -> Optional[str]:
        """Get station code by name"""
        return self.station_codes.get(name.lower())
    
    def heuristic(self, station1_code: str, station2_code: str) -> float:
        """
        Calculate heuristic (straight-line distance) between two stations
        """
        station1 = self.stations.get(station1_code)
        station2 = self.stations.get(station2_code)
        
        if not station1 or not station2:
            return 0.0
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        lat1, lon1 = math.radians(station1.lat), math.radians(station1.lon)
        lat2, lon2 = math.radians(station2.lat), math.radians(station2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = R * c
        
        # Convert to minutes (assuming 60 km/h average speed)
        return (distance_km / 60) * 60
    
    def get_all_stations(self) -> List[str]:
        """Get all station codes in the network"""
        return list(self.stations.keys())
    
    def get_all_station_names(self) -> List[str]:
        """Get all station names sorted alphabetically"""
        return sorted([station.name for station in self.stations.values()])
    
    def search_stations(self, query: str) -> List[Tuple[str, str]]:
        """Search for stations by name (returns list of (code, name) tuples)"""
        query = query.lower()
        results = []
        for code, station in self.stations.items():
            if query in station.name.lower():
                results.append((code, station.name))
        return sorted(results, key=lambda x: x[1])


# Test network loader
if __name__ == "__main__":
    print("=== Testing MRT Network Loader ===\n")
    
    # Test Today Mode
    print("Today Mode:")
    network_today = MRTNetwork(
        "data/nodes.csv",
        "data/edges.csv",
        mode="today"
    )
    print(f"Total stations: {len(network_today.stations)}")
    print(f"Sample stations: {list(network_today.stations.keys())[:5]}")
    
    # Test station search
    print("\nSearching for 'Changi':")
    results = network_today.search_stations("changi")
    for code, name in results:
        print(f"  {code}: {name}")
    
    # Test Future Mode
    print("\n\nFuture Mode:")
    network_future = MRTNetwork(
        "data/nodes.csv",
        "data/edges.csv",
        mode="future"
    )
    print(f"Total stations: {len(network_future.stations)}")
    
    # Check if T5 exists
    if "TE32/CR1" in network_future.stations:
        print(f" Terminal 5 found: {network_future.stations['TE32/CR1']}")
        print(f" Neighbors: {network_future.get_neighbors('TE32/CR1')}")
