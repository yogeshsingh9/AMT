"""
@author ys3276
"""
from mido import Message, MidiFile, MidiTrack
from mido import tick2second
from mido import second2tick
import numpy as np

default_tempo = 500000 # See https://mido.readthedocs.io/en/latest/midi_files.html
bottom_note = 21


"""
Writes a list of notes present at each time frame to a midi file. This allows us
to reconstruct the original midi file after labeling each frame with a note (or silence)
"""
def write_midi_multi(multi_notes, midi_file_name):

    midi_file = MidiFile(ticks_per_beat=10000)

    track = MidiTrack()
    midi_file.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))
    
    cur_time = 0
    prev_notes = np.full(89, False)
    for i, cur_notes in enumerate(multi_notes):
        cur_time += 0.005
        cur_ticks = second2tick(cur_time, midi_file.ticks_per_beat, default_tempo)
        for note_num in range(88):
            if cur_notes[note_num] == prev_notes[note_num]:
                continue
            else:
                if cur_notes[note_num]:
                    track.append(Message('note_on', note=int(bottom_note + note_num), velocity=127, time=int(cur_ticks)))
                    cur_time = 0
                    cur_ticks = 0
                else:
                    track.append(Message('note_off', note=int(bottom_note + note_num), velocity=127, time=int(cur_ticks)))
                    cur_time = 0
                    cur_ticks = 0
        prev_notes = cur_notes

    midi_file.save(midi_file_name)
