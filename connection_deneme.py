import serial
import time

# Set your serial port and baudrate
SERIAL_PORT = '/dev/rfcomm0'  # Bound Bluetooth port
BAUDRATE = 115200

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print("Connected to ESP32 with FluidNC")
except serial.SerialException as e:
    print("Failed to connect:", e)
    exit()

# Give it time to initialize
time.sleep(15)

# Send a command (e.g., move X axis 10 mm)
ser.write(b'G0 A100\n')  # Always use \n for newline
ser.flush()
time.sleep(1)
print("Command sent: G0 X10")
# Read response
while ser.in_waiting:
    response = ser.readline().decode().strip()
    print("Response:", response)

# Close connection
ser.close()
