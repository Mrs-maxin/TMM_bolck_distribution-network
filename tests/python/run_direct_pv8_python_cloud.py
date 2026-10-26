"""Resume analytic probability power flow from preserved cloud voltage parameters.

Accuracy comparisons are paused pending the user's confirmation of error logic.
Source-load scenarios, completed AC solutions and saved parameters are reused.
"""
import argparse
import platform
import traceback
from pathlib import Path
from complete_direct_pv8_analytic import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-openai-cloud',action='store_true')
    args=parser.parse_args()
    if not args.confirm_openai_cloud or platform.system()!='Linux':
        parser.error('Execution requires the declared OpenAI cloud Linux environment')
    try:
        run()
    except Exception:
        out=Path(__file__).resolve().parents[2]/'result/case33bw/direct_pv8_python_cloud_20261026/analytic_only'
        out.mkdir(parents=True,exist_ok=True)
        (out/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
