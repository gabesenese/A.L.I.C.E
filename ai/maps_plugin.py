"""
Maps and Places Plugin for A.L.I.C.E
Provides location-based services: nearby places, directions, etc.
Uses OpenStreetMap (free, no API key needed)
"""

import logging
import requests
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class MapsPlugin:
    """
    Maps and location services plugin.
    Finds nearby places, gives directions, provides location info.
    """
    
    def __init__(self):
        self.name = "MapsPlugin"
        self.version = "1.0.0"
        self.description = "Finds nearby places and provides location services"
        self.capabilities = ["nearby_places", "directions", "location_search"]
        self.enabled = True
    
    def initialize(self) -> bool:
        logger.info("🗺️ Maps plugin initialized")
        return True

    def get_info(self) -> Dict[str, str]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "capabilities": self.capabilities
        }
    
    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        if query is None:
            return False
        
        query_lower = query.lower()
        location_keywords = ['nearest', 'nearby', 'closest', 'find', 'where is', 'directions to', 'how to get to']
        place_types = ['supermarket', 'grocery', 'store', 'restaurant', 'cafe', 'bank', 'pharmacy', 'hospital', 'gas station']
        
        has_location_keyword = any(kw in query_lower for kw in location_keywords)
        has_place_type = any(pt in query_lower for pt in place_types)
        
        return has_location_keyword and has_place_type
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        """Find nearby places using OpenStreetMap"""
        try:
            # Ensure context is a dict
            if not isinstance(context, dict):
                context = {}
            
            # Get user location
            user_location = context.get('location')
            user_city = context.get('city')
            
            if not user_city:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_location",
                        "message_code": "maps:no_location"
                    }
                }
            
            # Extract what they're looking for
            query_lower = query.lower()
            place_type = self._extract_place_type(query_lower)
            
            if not place_type:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_place_type",
                        "message_code": "maps:no_place_type"
                    }
                }
            
            # Search for places using Overpass API (OpenStreetMap)
            places = self._find_nearby_places(user_city, place_type)
            
            if not places:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "place_type": place_type,
                        "location": user_city,
                        "message_code": "maps:no_results"
                    }
                }
            
            # Return data for A.L.I.C.E to generate response
            return {
                "success": True,
                "response": None,
                "data": {
                    "place_type": place_type,
                    "location": user_city,
                    "places": places[:5],  # Top 5 results
                    "plugin_type": "maps",
                    "count": len(places),
                    "message_code": "maps:found"
                }
            }
            
        except Exception as e:
            logger.error(f"Maps plugin error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "maps:error"
                }
            }
    
    def _extract_place_type(self, query: str) -> Optional[str]:
        """Extract what type of place user is looking for"""
        place_mappings = {
            'supermarket': 'supermarket',
            'grocery': 'supermarket',
            'store': 'shop',
            'restaurant': 'restaurant',
            'cafe': 'cafe',
            'coffee': 'cafe',
            'bank': 'bank',
            'atm': 'atm',
            'pharmacy': 'pharmacy',
            'hospital': 'hospital',
            'gas station': 'fuel',
            'gas': 'fuel',
            'hotel': 'hotel',
            'park': 'park'
        }
        
        for key, value in place_mappings.items():
            if key in query:
                return value
        
        return None
    
    def _find_nearby_places(self, city: str, place_type: str, radius_km: int = 5) -> List[Dict]:
        """
        Find nearby places using Nominatim (OpenStreetMap geocoding) and Overpass API.
        Free, no API key required.
        """
        try:
            # Step 1: Get coordinates for the city
            coords = self._geocode_city(city)
            if not coords:
                logger.warning(f"Could not geocode city: {city}")
                return []
            
            lat, lon = coords
            
            # Step 2: Search for places near those coordinates using Overpass API
            # Overpass query to find places within radius
            overpass_url = "https://overpass-api.de/api/interpreter"
            
            # Build Overpass QL query
            query = f"""
            [out:json][timeout:10];
            (
              node["shop"="{place_type}"](around:{radius_km * 1000},{lat},{lon});
              way["shop"="{place_type}"](around:{radius_km * 1000},{lat},{lon});
              node["amenity"="{place_type}"](around:{radius_km * 1000},{lat},{lon});
              way["amenity"="{place_type}"](around:{radius_km * 1000},{lat},{lon});
            );
            out body 20;
            """
            
            response = requests.post(overpass_url, data={"data": query}, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                
                places = []
                for elem in elements:
                    name = elem.get('tags', {}).get('name', 'Unnamed')
                    place_lat = elem.get('lat')
                    place_lon = elem.get('lon')
                    
                    if name != 'Unnamed' and place_lat and place_lon:
                        # Calculate approximate distance (rough)
                        dist_km = self._calculate_distance(lat, lon, place_lat, place_lon)
                        
                        places.append({
                            'name': name,
                            'distance_km': round(dist_km, 1),
                            'lat': place_lat,
                            'lon': place_lon
                        })
                
                # Sort by distance
                places.sort(key=lambda x: x['distance_km'])
                return places
            
        except Exception as e:
            logger.error(f"Overpass API error: {e}")
        
        return []
    
    def _geocode_city(self, city: str) -> Optional[tuple]:
        """Get lat/lon for a city using Nominatim"""
        try:
            url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
            headers = {'User-Agent': 'A.L.I.C.E/1.0'}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
        
        return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate approximate distance in km using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c  # Earth radius in km
        
        return km
    
    def shutdown(self):
        logger.info("🗺️ Maps plugin shutdown")
