"""Executes doctests for all files listed in 'doctest_files'.

The file should contain the following three lines at the end
of the file to enable execution of doctests:

if __name__ == "__main__":
    import doctest
    doctest.testmod()

Exit code 0 on success, -1 on error.
"""

import sys
import subprocess

retcode = 0

# all files that have doctests
doctest_files = [
    "elfi/core.py",
    "elfi/storage.py",
    ]

for f in doctest_files:
    out = subprocess.check_output(["python", f], universal_newlines=True)
    if len(out) != 0:
        print(out)
        retcode = -1
    else:
        print("{} .. ok".format(f))

sys.exit(retcode)
