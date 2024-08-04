import os,sys

working_dir = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(working_dir):
    sys.path.append(working_dir)

