from dataclasses import dataclass, field
from tkinter import BooleanVar
from typing import List, Dict, Optional
import serial
import time

# sudo rfcomm release 0
# sudo rfcomm bind 0 6C:C8:40:06:C4:F6 1
# picocom -b 115200 /dev/rfcomm0



@dataclass
class MotorCommand:
    solution_str: str
    command: Optional[List[str]] = None
    solution_seq: Optional[List[str]] = None
    motors: Dict[str, str] = field(default_factory=lambda: {
        'D': 'X',  # top
        'U': 'Y',  # bottom
        'L': 'Z',  # left
        'R': 'A',  # right
        'F': 'B',  # back
        'B': 'C'   # front
    })

    # 'U': 'X', 'D': 'Y'
    serial_port: str = '/dev/rfcomm0'
    serial_baudrate: int = 115200
    serial_timeout: int = 1
    serial_connection: Optional[serial.Serial] = field(init=False, default=None)

    def __post_init__(self):
        try:
            self.serial_connection = serial.Serial(
                self.serial_port,
                self.serial_baudrate,
                timeout=self.serial_timeout
            )
            print(f"Connected to {self.serial_port}")
            # self.send_command('$I')  # firmware info
            time.sleep(0.15)
            self.send_command('G91')  # relative mode

        except serial.SerialException as e:
            print(f"Failed to connect to serial port: {e}")
            self.serial_connection = None

    def convert_to_sequence(self) -> None:
        self.solution_seq = self.solution_str.split(' ')

    def generate_motor_commands(self) -> Optional[List[str]]:
        rotation_map = {'1': '1', '2': '2', '3': '-1'}
        self.command = []

        for move in self.solution_seq:
            if len(move) >= 2:
                face = move[0]
                number = move[1]
                motor = self.motors.get(face)
                degrees = rotation_map.get(number)
                if motor and degrees:
                    self.command.append(f"G0{motor}{degrees}")
                else:
                    self.command.append(f"Invalid {move}")
            else:
                self.command.append(f"Invalid {move}")

    def send_command(self, cmd: str) -> None:
        if not cmd or self.serial_connection is None:
            print("No valid command or serial connection.")
            return

        self.serial_connection.write((cmd + '\n').encode())
        time.sleep(0.15)
        while self.serial_connection.in_waiting:
            print(self.serial_connection.readline().decode().rstrip())

    def send_sequence(self) -> None:
        if self.command is None or self.serial_connection is None:
            print("No commands or serial connection.")
            return

        for cmd in self.command:
            self.send_command(cmd)

        self.serial_connection.close()
        print("Serial connection closed.")



# Example usage
if __name__ == "__main__":
    data = MotorCommand("L1 L1 L3")
    data.convert_to_sequence()
    print("Solution sequence:", data.solution_seq)
    data.generate_motor_commands()
    print("Generated motor commands:", data.command)

    data.send_sequence()