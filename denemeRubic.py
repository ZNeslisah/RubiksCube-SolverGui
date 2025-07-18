from sendToEsp import MotorCommand


full_solution = 'R1 D3 F3 B1'
# R3 B3 D3 R2 L1 F3 R2 U1 F1 B1 U2 R2 D3 B2 D2 R2 U1 B2 U3
data = MotorCommand(full_solution)

data.convert_to_sequence()
data.generate_motor_commands()
data.send_sequence()