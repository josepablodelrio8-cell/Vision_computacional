import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"Puerto : {port.device}")
    print(f"Desc   : {port.description}")
    print(f"HWID   : {port.hwid}")
    print("---")