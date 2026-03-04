# ML-Automixer-Project

SETUP:
metadata.csv, SALAMI_iTunes_library, and annotations folder (required for you to find in the SALAMI dataset cause its big and i haven't pulled this repo yet) from [SALAMI](https://github.com/DDMAL/salami-data-public/tree/master). these files should all be on the same folder level as the python script.

the joincsv.py file is just a little messing around to figure out how to join the csvs (i mean basically SQL in Python)

more info about how to run the main file are in the arguments/code, for base run: python3 main.py --csv metadata.csv --no-shuffle
