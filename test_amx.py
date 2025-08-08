import ctypes
import os

lib = ctypes.CDLL('./libamx_demo.so')

# Call the AMX demo
result = lib.run_amx_demo()
if result == 0:
    print("✅ AMX demo ran successfully")
else:
    print("❌ AMX demo failed")
