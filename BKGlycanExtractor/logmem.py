
import sys
import psutil
import inspect

def getmemmb():
    p = psutil.Process()
    mi = p.memory_info()
    vmem = mi.vms; rmem = mi.rss
    for cp in p.children():
        mi = cp.memory_info()
        vmem += mi.vms; rmem += mi.rss
    return vmem/1024**2, rmem/1024**2

def logmem():
    vmemmb,rmemmb = getmemmb()
    print(inspect.stack()[1][1],":",inspect.stack()[1][2],":",
          inspect.stack()[1][3],":",
          "virt. memory %.0f MB, resident mem. %.0f MB."%(vmemmb,rmemmb),
          file=sys.stderr)

