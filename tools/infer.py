"""Compatibility launcher for projects/dualtopo."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/"projects/dualtopo"
sys.path.insert(0,str(ROOT))
if __name__=="__main__":runpy.run_path(str(ROOT/"tools"/'infer.py'),run_name="__main__")
