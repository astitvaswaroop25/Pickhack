import threading
import time
import random

class SensorReader:
    def __init__(self, port="MOCK", baud=9600):
        self.counts = {"lane1": 0, "lane2": 0}
        self.running = True
        # Serial connection is commented out for mocking
        # self.serial_conn = serial.Serial(port, baud, timeout=1)

    def start(self):
        # Starts the background thread for the "fake" data loop
        thread = threading.Thread(target=self._mock_read_loop, daemon=True)
        thread.start()

    def _mock_read_loop(self):
        """Simulates vehicles passing by every 5-10 seconds."""
        while self.running:
            time.sleep(random.randint(5, 10)) 
            lane = random.choice(["lane1", "lane2"])
            self.counts[lane] += 1
            print(f"MOCK DATA: {lane} incremented to {self.counts[lane]}")

    def get_counts(self):
        # Returns the current fake counts to the dashboard
        return self.counts.copy()