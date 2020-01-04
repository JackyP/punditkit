import os
import sys
import streamlit.bootstrap as bootstrap
from pathlib import Path


def run():
    path = (Path(__file__).parent / "modelling.py").absolute()
    arg1 = sys.argv[1]
    bootstrap.run(path, command_line=os.getcwd(), args={arg1})
