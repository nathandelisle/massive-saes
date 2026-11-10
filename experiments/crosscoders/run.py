"""Run the four crosscoder comparisons in Appendix M."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("RX_LOCAL", "1")
from ext.coders_job import main

if __name__ == "__main__":
    sys.argv.insert(1, "C")
    main()
