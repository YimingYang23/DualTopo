"""Shared RATopo entry point; select Subset A or B explicitly."""
import argparse,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--subset',required=True,choices=['A','B','a','b'])
if '--help' in sys.argv and '--subset' not in sys.argv:
    p.description='Shared RATopo CLI. Select --subset A or B; add --help after selecting a subset for task options.'
    p.print_help();sys.exit(0)
a,remaining=p.parse_known_args()
subset=a.subset.lower()
backend=ROOT/'backends'/('subset_'+subset)
sys.path.insert(0,str(backend))
if '--output' not in remaining:remaining += ['--output',str(ROOT/'weights'/('ratopo_subset_'+subset+'.safetensors'))]
sys.argv=[str(backend/"tools"/'download_weights.py')]+remaining
runpy.run_path(sys.argv[0],run_name="__main__")
