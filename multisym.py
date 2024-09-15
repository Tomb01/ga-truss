import subprocess
import os
import sys

sim_name = os.getcwd()+"/.trash/39bar-4"

print("block 1")
subprocess.check_output([sys.executable, "simulate.py", sim_name, str(3), str(0.3), str(0.1), str(0.01)], shell=True)
print("block 1")
subprocess.check_output([sys.executable, "simulate.py", sim_name, str(3), str(0.2), str(0.1), str(0.01)], shell=True)
print("block 1")
subprocess.check_output([sys.executable, "simulate.py", sim_name, str(3), str(0.25), str(0.1), str(0.01)], shell=True)