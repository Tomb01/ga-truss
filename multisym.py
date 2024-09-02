import subprocess
import os
import sys

sim_name = os.getcwd()+"/.trash/10bar-area-1"

subprocess.check_output([sys.executable, "simulate.py", sim_name, str(10)], shell=True)