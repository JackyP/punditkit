import os
import sys
import streamlit.bootstrap as bootstrap


def run():
    path = os.path.abspath(sys.argv[0].replace("run.py", "modelling.py"))
    print(path)
    bootstrap.run(path, command_line="", args="")
