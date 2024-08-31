import subprocess
import os
import sys

sim_name = os.getcwd()+"/.trash/10bar4"

for m in range(0,3):
    mutatio_ratio = m/10
    subprocess.check_output([sys.executable, "simulate.py", sim_name, str(0), str(mutatio_ratio), str(0.1)], shell=True)