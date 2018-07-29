import json
import bz2
import sys

with bz2.BZ2File(sys.argv[1]) as matches_file:
    for line in matches_file:
        match = json.loads(line.decode())

        # Обработка матча
        print(match)
        break

