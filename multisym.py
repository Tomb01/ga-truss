import subprocess
import os
import sys

sim_name = os.getcwd()+"/.trash/10bar5-def"

subprocess.check_output([sys.executable, "simulate.py", sim_name, str(0), str(0.2), str(0.1)], shell=True)
subprocess.check_output([sys.executable, "simulate.py", sim_name, str(0.1), str(0.2), str(0.1)], shell=True)