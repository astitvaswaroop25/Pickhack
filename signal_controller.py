import time

class TrafficSignalController:
    def __init__(self):
        self.base_green_time = 30
        self.emergency_mode = False

    def update(self, analysis: dict, sensor_counts: dict):
        if analysis.get("emergency_priority"):
            self.emergency_mode = True
            return {"action": "emergency_clear", "message": "Emergency vehicle approaching!"}
        
        pedestrians = analysis.get("pedestrians", [])
        if any(p.get("crossing") for p in pedestrians):
            return {"action": "pedestrian_crossing", "message": "Pedestrian crossing active"}
            
        density = analysis.get("traffic_density", "medium")
        multiplier = {"low": 0.7, "medium": 1.0, "high": 1.3, "gridlock": 1.5}
        green_time = self.base_green_time * multiplier.get(density, 1.0)
        return {"action": "adaptive", "message": f"Density: {density} - green: {green_time:.0f}s"}