"""
MRT Network Graph Representation
Supports both "Today Mode" and "Future Mode" networks
Based on real Singapore MRT data and LTA's July 2025 TELe/CRL announcement
"""

import math
from typing import Dict, List, Tuple, Optional

class MRTStation:
    """Represents an MRT station with coordinates"""
    def __init__(self, code: str, name: str, line: str, lat: float, lon: float):
        self.code = code
        self.name = name
        self.line = line
        self.lat = lat
        self.lon = lon
    
    def __repr__(self):
        return f"{self.code} {self.name}"
    
    def __eq__(self, other):
        return self.code == other.code if isinstance(other, MRTStation) else False
    
    def __hash__(self):
        return hash(self.code)


class MRTNetwork:
    """
    MRT Network Graph with both Today and Future modes
    
    Edge weights are based on travel time in minutes:
    - Base travel time: ~2 minutes between adjacent stations (real-world average)
    - Transfer penalty: +3 minutes for line changes (includes walking + waiting)
    - Crowding penalty: +0-2 minutes depending on peak hours and station load
    
    Coordinates are approximate based on Singapore geography
    """
    
    def __init__(self, mode="today"):
        """
        Initialize MRT network
        
        Args:
            mode: "today" for current network, "future" for TELe/CRL network
        """
        self.mode = mode
        self.stations: Dict[str, MRTStation] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}  # {station_code: [(neighbor_code, travel_time), ...]}
        self._initialize_stations()
        self._initialize_connections()
    
    def _initialize_stations(self):
        """Initialize station data with approximate coordinates"""
        # East-West Line (EWL) stations
        ewl_stations = [
            ("EW4", "Tanah Merah", "EWL", 1.3274, 103.9464),
            ("EW5", "Expo", "EWL", 1.3354, 103.9614),
            ("CG2", "Changi Airport", "EWL", 1.3573, 103.9888),
            ("EW7", "Simei", "EWL", 1.3420, 103.9534),
            ("EW1", "Pasir Ris", "EWL", 1.3730, 103.9492),
            ("EW6", "Bedok", "EWL", 1.3240, 103.9301),
            ("EW8", "Kembangan", "EWL", 1.3208, 103.9123),
            ("EW9", "Eunos", "EWL", 1.3196, 103.9034),
            ("EW10", "Paya Lebar", "EWL", 1.3177, 103.8927),
            ("EW11", "Aljunied", "EWL", 1.3164, 103.8819),
            ("EW12", "Kallang", "EWL", 1.3114, 103.8716),
            ("EW13", "Lavender", "EWL", 1.3074, 103.8631),
            ("EW14", "Bugis", "EWL", 1.3005, 103.8558),
            ("EW15", "City Hall", "EWL", 1.2931, 103.8520),
            ("EW16", "Raffles Place", "EWL", 1.2837, 103.8514),
            ("EW17", "Tanjong Pagar", "EWL", 1.2764, 103.8457),
            ("EW18", "Outram Park", "EWL", 1.2801, 103.8399),
        ]
        
        # Downtown Line (DTL) stations
        dtl_stations = [
            ("DT35", "Expo", "DTL", 1.3354, 103.9614),
            ("DT36", "Xilin", "DTL", 1.3474, 103.9518),
            ("DT37", "Sungei Bedok", "DTL", 1.3390, 103.9434),
            ("CG3", "Upper Changi", "DTL", 1.3414, 103.9616),
            ("DT32", "Tampines East", "DTL", 1.3564, 103.9554),
            ("DT33", "Tampines West", "DTL", 1.3456, 103.9387),
            ("DT34", "Bedok Reservoir", "DTL", 1.3357, 103.9332),
            ("DT25", "Bedok North", "DTL", 1.3351, 103.9185),
            ("DT26", "Kaki Bukit", "DTL", 1.3349, 103.9085),
            ("DT27", "Ubi", "DTL", 1.3302, 103.8996),
            ("DT28", "MacPherson", "DTL", 1.3267, 103.8897),
            ("DT29", "Mattar", "DTL", 1.3265, 103.8833),
            ("DT23", "Geylang Bahru", "DTL", 1.3210, 103.8716),
            ("DT22", "Bendemeer", "DTL", 1.3140, 103.8624),
            ("DT21", "Jalan Besar", "DTL", 1.3055, 103.8553),
            ("DT20", "Bencoolen", "DTL", 1.2985, 103.8506),
            ("DT19", "Dhoby Ghaut", "DTL", 1.2990, 103.8458),
            ("DT18", "Fort Canning", "DTL", 1.2909, 103.8447),
            ("DT19A", "Chinatown", "DTL", 1.2844, 103.8437),
            ("DT17", "Outram Park", "DTL", 1.2801, 103.8399),
        ]
        
        # Thomson-East Coast Line (TEL) stations
        tel_stations = [
            ("TE31", "Sungei Bedok", "TEL", 1.3390, 103.9434),
            ("TE30", "Bedok South", "TEL", 1.3210, 103.9448),
            ("TE29", "Bayshore", "TEL", 1.3116, 103.9496),
            ("TE28", "Siglap", "TEL", 1.3117, 103.9274),
            ("TE27", "Marine Terrace", "TEL", 1.3065, 103.9146),
            ("TE26", "Marine Parade", "TEL", 1.3021, 103.9063),
            ("TE25", "Tanjong Katong", "TEL", 1.3003, 103.8943),
            ("TE24", "Katong Park", "TEL", 1.3013, 103.8850),
            ("TE23", "Tanjong Rhu", "TEL", 1.2946, 103.8773),
            ("TE22", "Gardens by the Bay", "TEL", 1.2810, 103.8646),
            ("TE20", "Marina Bay", "TEL", 1.2767, 103.8540),
            ("TE19", "Shenton Way", "TEL", 1.2785, 103.8498),
            ("TE18", "Maxwell", "TEL", 1.2807, 103.8439),
            ("TE17", "Outram Park", "TEL", 1.2801, 103.8399),
        ]
        
        # Add stations to the network
        for code, name, line, lat, lon in ewl_stations + dtl_stations + tel_stations:
            self.stations[code] = MRTStation(code, name, line, lat, lon)
        
        # Future Mode: Add Terminal 5 station
        if self.mode == "future":
            self.stations["TE32/CR1"] = MRTStation("TE32/CR1", "Changi Terminal 5", "TEL/CRL", 1.3550, 103.9950)
            # Convert EWL Changi Airport branch to TEL
            self.stations["CG2"].line = "TEL"
            self.stations["EW5"].line = "TEL"  # Expo becomes TEL
            self.stations["EW4"].line = "TEL/EWL"  # Tanah Merah becomes interchange
    
    def _initialize_connections(self):
        """Initialize edges with travel times based on mode"""
        
        if self.mode == "today":
            self._add_today_mode_connections()
        else:
            self._add_future_mode_connections()
    
    def _add_today_mode_connections(self):
        """Add connections for current network (Today Mode)"""
        
        # EWL connections (main line)
        ewl_mainline = [
            ("EW10", "EW11"), ("EW11", "EW12"), ("EW12", "EW13"),
            ("EW13", "EW14"), ("EW14", "EW15"), ("EW15", "EW16"),
            ("EW16", "EW17"), ("EW17", "EW18")
        ]
        
        # EWL Changi Airport branch
        ewl_changi_branch = [
            ("EW4", "EW5"), ("EW5", "CG2")
        ]
        
        # EWL Pasir Ris branch
        ewl_pasir_ris_branch = [
            ("EW4", "EW7"), ("EW7", "EW1"), 
            ("EW6", "EW8"), ("EW8", "EW9"), ("EW9", "EW10")
        ]
        
        # Tanah Merah branching
        self._add_bidirectional_edge("EW4", "EW6", 2.0)  # Tanah Merah to Bedok
        
        # DTL connections
        dtl_connections = [
            ("DT35", "DT36"), ("DT35", "CG3"), ("DT37", "DT35"),  # Expo branches
            ("DT32", "DT33"), ("DT33", "DT34"), ("DT34", "DT25"),
            ("DT25", "DT26"), ("DT26", "DT27"), ("DT27", "DT28"),
            ("DT28", "DT29"), ("DT23", "DT22"), ("DT22", "DT21"),
            ("DT21", "DT20"), ("DT20", "DT19"), ("DT19", "DT18"),
            ("DT18", "DT19A"), ("DT19A", "DT17")
        ]
        
        # TEL connections
        tel_connections = [
            ("TE31", "TE30"), ("TE30", "TE29"), ("TE29", "TE28"),
            ("TE28", "TE27"), ("TE27", "TE26"), ("TE26", "TE25"),
            ("TE25", "TE24"), ("TE24", "TE23"), ("TE23", "TE22"),
            ("TE22", "TE20"), ("TE20", "TE19"), ("TE19", "TE18"),
            ("TE18", "TE17")
        ]
        
        # Add all connections with 2-minute base travel time
        for conn in ewl_mainline + ewl_changi_branch + ewl_pasir_ris_branch + dtl_connections + tel_connections:
            self._add_bidirectional_edge(conn[0], conn[1], 2.0)
        
        # Interchange stations (with transfer penalty)
        # Expo: EWL <-> DTL
        self._add_bidirectional_edge("EW5", "DT35", 3.0)  # Transfer time
        
        # Outram Park: EWL <-> DTL <-> TEL
        self._add_bidirectional_edge("EW18", "DT17", 3.0)
        self._add_bidirectional_edge("EW18", "TE17", 3.0)
        self._add_bidirectional_edge("DT17", "TE17", 3.0)
    
    def _add_future_mode_connections(self):
        """Add connections for future network with TELe/CRL (Future Mode)"""
        
        # EWL connections (main line - unchanged)
        ewl_mainline = [
            ("EW10", "EW11"), ("EW11", "EW12"), ("EW12", "EW13"),
            ("EW13", "EW14"), ("EW14", "EW15"), ("EW15", "EW16"),
            ("EW16", "EW17"), ("EW17", "EW18")
        ]
        
        # EWL Pasir Ris branch (unchanged)
        ewl_pasir_ris_branch = [
            ("EW6", "EW8"), ("EW8", "EW9"), ("EW9", "EW10"),
            ("EW7", "EW1")
        ]
        
        # Tanah Merah branching (EWL side)
        self._add_bidirectional_edge("EW4", "EW6", 2.0)
        self._add_bidirectional_edge("EW4", "EW7", 2.0)
        
        # TELe: Sungei Bedok -> T5 -> Changi Airport -> Tanah Merah (converted to TEL)
        tel_extension = [
            ("TE31", "TE32/CR1"),  # Sungei Bedok to T5
            ("TE32/CR1", "CG2"),    # T5 to Changi Airport (now TEL)
            ("CG2", "EW5"),          # Changi Airport to Expo (now TEL)
            ("EW5", "EW4")           # Expo to Tanah Merah (interchange)
        ]
        
        # DTL connections (mostly unchanged except branches at Expo)
        dtl_connections = [
            ("DT37", "DT35"), ("DT35", "DT36"), ("DT35", "CG3"),  # Expo branches
            ("DT32", "DT33"), ("DT33", "DT34"), ("DT34", "DT25"),
            ("DT25", "DT26"), ("DT26", "DT27"), ("DT27", "DT28"),
            ("DT28", "DT29"), ("DT23", "DT22"), ("DT22", "DT21"),
            ("DT21", "DT20"), ("DT20", "DT19"), ("DT19", "DT18"),
            ("DT18", "DT19A"), ("DT19A", "DT17")
        ]
        
        # TEL connections (original TEL route)
        tel_connections = [
            ("TE31", "TE30"), ("TE30", "TE29"), ("TE29", "TE28"),
            ("TE28", "TE27"), ("TE27", "TE26"), ("TE26", "TE25"),
            ("TE25", "TE24"), ("TE24", "TE23"), ("TE23", "TE22"),
            ("TE22", "TE20"), ("TE20", "TE19"), ("TE19", "TE18"),
            ("TE18", "TE17")
        ]
        
        # Add all connections with 2-minute base travel time
        for conn in ewl_mainline + ewl_pasir_ris_branch + tel_extension + dtl_connections + tel_connections:
            self._add_bidirectional_edge(conn[0], conn[1], 2.0)
        
        # Interchange stations (with transfer penalty)
        # Tanah Merah: EWL <-> TEL (major interchange in future)
        self._add_bidirectional_edge("EW4", "EW4", 3.0)  # Transfer within station
        
        # Expo: DTL <-> TEL
        self._add_bidirectional_edge("EW5", "DT35", 3.0)
        
        # Outram Park: EWL <-> DTL <-> TEL (triple interchange)
        self._add_bidirectional_edge("EW18", "DT17", 3.0)
        self._add_bidirectional_edge("EW18", "TE17", 3.0)
        self._add_bidirectional_edge("DT17", "TE17", 3.0)
    
    def _add_bidirectional_edge(self, station1: str, station2: str, weight: float):
        """Add bidirectional edge between two stations"""
        if station1 not in self.edges:
            self.edges[station1] = []
        if station2 not in self.edges:
            self.edges[station2] = []
        
        self.edges[station1].append((station2, weight))
        self.edges[station2].append((station1, weight))
    
    def get_neighbors(self, station_code: str) -> List[Tuple[str, float]]:
        """Get neighbors of a station with travel times"""
        return self.edges.get(station_code, [])
    
    def get_station(self, station_code: str) -> Optional[MRTStation]:
        """Get station object by code"""
        return self.stations.get(station_code)
    
    def heuristic(self, station1_code: str, station2_code: str) -> float:
        """
        Calculate heuristic (straight-line distance) between two stations
        
        Uses Haversine formula for geographic distance, then converts to
        approximate travel time assuming average speed of 60 km/h
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
        return (distance_km / 60) * 60  # (km / km/h) * 60 min/h
    
    def get_all_stations(self) -> List[str]:
        """Get all station codes in the network"""
        return list(self.stations.keys())


# Test the network
if __name__ == "__main__":
    print("=== Today Mode ===")
    network_today = MRTNetwork(mode="today")
    print(f"Total stations: {len(network_today.stations)}")
    print(f"Changi Airport neighbors: {network_today.get_neighbors('CG2')}")
    
    print("\n=== Future Mode ===")
    network_future = MRTNetwork(mode="future")
    print(f"Total stations: {len(network_future.stations)}")
    print(f"T5 neighbors: {network_future.get_neighbors('TE32/CR1')}")
    print(f"Changi Airport neighbors: {network_future.get_neighbors('CG2')}")
