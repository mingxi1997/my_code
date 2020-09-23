import os
import time
a=subprocess.Popen('mitmproxy -w test.log', stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
time.sleep(50)
os.killpg(os.getpgid(a.pid), signal.SIGTERM)
