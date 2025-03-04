#!/usr/bin/env python3

import os

n = int(input("Entrez un nombre : "))

for i in range(n + 1):
	with open(f"{i}-neural_syle.py", "w") as f:
		f.write("#!/usr/bin/env python3\n")
	os.chmod(f"{i}-neural_syle.py", 0o755)
