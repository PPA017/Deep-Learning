import numpy as np
import os
import time
import functools
import torch
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write

import subprocess
import platform

import utils

assert torch.cuda.is_available(),"[FAIL] NO GPU FOUND"

def play_song(midi_file):
    """
    Play a MIDI file using TiMidity++ on Windows, or default method on Linux/Mac.
    """
    if platform.system() == "Windows":
        # Full path to your timidity.exe
        timidity_path = r"C:\Users\pusca\Desktop\Uni\stuff\TiMidity++-2.15.0\timidity.exe"
        subprocess.run([timidity_path, midi_file])
    else:
        # Original Linux/Mac command
        subprocess.run(["chmod", "+x", "./play_midi.sh"])
        subprocess.run(["./play_midi.sh", midi_file])
    

songs = utils.lab1.load_training_data()

example_song = songs[0]
#print("\nExample song: ")
#print(example_song)

play_song(example_song)