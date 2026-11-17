"""Expose Ubuntu ROS/OpenCV bindings after the CUDA venv's own packages."""
from pathlib import Path
import runpy
import sys

sys.path.append('/usr/lib/python3/dist-packages')
runpy.run_path(str(Path(__file__).with_name('inference_node.py')), run_name='__main__')
