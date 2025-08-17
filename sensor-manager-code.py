import RPi.GPIO as GPIO
import time
import Adafruit_DHT

class SensorManager:
    def __init__(self):
        # === GPIO Pin Configuration ===
        self.PIR_PIN = 17       # Motion sensor
        self.IR_PIN = 27        # IR gesture sensor
        self.TRIG = 23          # Ultrasonic sensor TRIG
        self.ECHO = 24          # Ultrasonic sensor ECHO
        self.DHT_PIN = 4        # DHT temperature sensor pin
        self.DHT_SENSOR = Adafruit_DHT.DHT22  # Change to DHT11 if using that sensor
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PIR_PIN, GPIO.IN)
        GPIO.setup(self.IR_PIN, GPIO.IN)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        
        # Debounce variable
        self.last_gesture_time = 0
        
        # Initialize ultrasonic sensor
        GPIO.output(self.TRIG, False)
        time.sleep(0.5)  # Let the sensor settle
        
    def read_pir(self):
        """Read PIR motion sensor"""
        return GPIO.input(self.PIR_PIN) == 1
    
    def read_ir(self):
        """Read IR gesture sensor"""
        return GPIO.input(self.IR_PIN) == 0  # Active LOW
    
    def measure_distance(self):
        """Measure distance with ultrasonic sensor"""
        try:
            GPIO.output(self.TRIG, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG, False)
            
            pulse_start = time.time()
            pulse_end = time.time()
            timeout = time.time() + 0.1  # 100ms timeout
            
            # Wait for echo to go high or timeout
            while GPIO.input(self.ECHO) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return 400  # Return max range if timeout
            
            # Wait for echo to go low or timeout
            while GPIO.input(self.ECHO) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return 400  # Return max range if timeout
            
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Speed of sound = 343m/s
            distance = round(distance, 2)
            
            # Clamp to reasonable range
            if distance > 400:
                distance = 400
            
            return distance
        except:
            return 400  # Return max range on error
    
    def get_temperature(self):
        """Get temperature and humidity from DHT sensor"""
        humidity, temperature = Adafruit_DHT.read_retry(self.DHT_SENSOR, self.DHT_PIN)
        if humidity is not None and temperature is not None:
            return round(temperature, 1), round(humidity, 1)
        else:
            return None, None
    
    def read_all_sensors(self):
        """Read all sensors and return data dictionary"""
        distance = self.measure_distance()
        return {
            'pir_detected': self.read_pir(),
            'ir_detected': self.read_ir(),
            'distance': distance,
            'all_operational': True
        }
    
    def cleanup(self):
        """Cleanup GPIO pins"""
        GPIO.cleanup()
