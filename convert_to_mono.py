#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:43:42 2019

@author: yogesh
"""
from os import listdir
from pydub import AudioSegment
from os.path import  join


def convert_to_mono(src_dir, dest_dir):
    for i, wav_file in enumerate(sorted(listdir(src_dir))):
        if not ".wav" in wav_file:
            continue
        sound = AudioSegment.from_wav(join(src_dir, wav_file))
        sound = sound.set_channels(1)
        sound.export(join(dest_dir, wav_file), format="wav")