import time

class FailSafeWatchdog:
    def __init__(self, timeout_seconds=5):
        self.timeout = timeout_seconds
        self.last_inference_time = time.time()
        self.alert_triggered = False

    def heartbeat(self):
        self.last_inference_time = time.time()
        self.alert_triggered = False

    def monitor(self):
        while True:
            if time.time() - self.last_inference_time > self.timeout and not self.alert_triggered:
                print("[WATCHDOG ALERT] Inference stalled! Taking recovery action...")
                self.alert_triggered = True
            time.sleep(1)