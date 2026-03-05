# ML-Automixer-Project

SETUP:
make sure to download the entire repo from [SALAMI](https://github.com/DDMAL/salami-data-public/tree/master). when running main, you are going to need the file path to the 
metadata folder like my example below:
for base run: python main.py --csv /Users/ninjadare/Downloads/salami-data-public-master/metadata --n 5
--n allows us to cap how many things to download
--txt was not touched at all by me haha idk what it does
see annotations folder for all the raw inputs, we use parsed functions

you can delete audio, spectograms and manifest if you want a clean slate, because the code will skip things already processed
i was thinking to git ignore those three files/folders but it might be worth it to push through

# todo
needs to validate length of song downloaded from 
delete a ton of the commented out and redundant code not being used anymore
the manifest csv is HUGE maybe we can think on how to reduce it

# for venv (git ignored)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt