""" Use: python cleaner.py 'ANY-CMD' """

import psutil, sys

pids = list()
for process in psutil.process_iter():
    cmd = " ".join(process.cmdline())
    
    if sys.argv[1] in cmd and not cmd.startswith("python cleaner.py"):
        pids.append(process.pid)

for pid in pids:
    print(f"Terminate: {pid}")
    p = psutil.Process(pid)
    p.terminate()