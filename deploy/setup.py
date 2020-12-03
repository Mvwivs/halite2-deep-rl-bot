
import sys
from cx_Freeze import setup, Executable

build_exe_options = {"packages": ["os", "models"],}

base = None

setup(  name = "MlrlBot",
        version = "0.1",
        description = "My Bot!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("MlrlBotTest.py", base=base)])
