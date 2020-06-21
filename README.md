# voice-to-notes
Taicheng Song's voice to notes project

This project take a .wav audio file of human singing, and output an optimal beats per minute, beats per note, frequency for of note, and duration of a beat(in ms).

The return format is a dictionary with: {"interval_length", "beats_per_note", "frequencies", "bpm"} as keys

Use record=True if you wish the program to record your singing into .wav file.


# Example use case:

execute()

Basically runs by default as: 
execute(wav_filename="temps/output.wav", txt_filename="temps/output.wav", step_size=10, use_txt_input=False, record=False, audio_output=False, graph=False, style="WEST", scale=None, verbose=False, mute=False)
It will execute on "temps/output.wav". If the file does not exist, it will print an error. 
    
