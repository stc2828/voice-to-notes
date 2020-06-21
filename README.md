# Voice-to-notes
Taicheng Song's voice to notes project

This project take a .wav audio file of human singing, and output an optimal beats per minute, beats per note, frequency for of note, and duration of a beat(in ms).

The return format is a dictionary with: {"interval_length", "beats_per_note", "frequencies", "bpm"} as keys

Use record=True if you wish the program to record your singing into .wav file.

This module find frequency with crepe, which on average take longer to run then the duration of the input sound file. 


### Tips
Instead of sining the lyrics, try to sin with simple, burst of sound such as: "Ding, Ding, Ding", "Dong, Dong, Dong", etc. For better pickup accuracy.


### Example use case:
###### Case 1:
execute()

Will run every parameter on default as: 
execute(wav_filename="temps/output.wav", txt_filename="temps/output.wav", step_size=10, use_txt_input=False, record=False, audio_output=False, graph=False, style="WEST", scale=None, verbose=False, mute=False)
It will execute on "temps/output.wav". If the file does not exist, it will print an error. 
    
###### Case 2:
execute(record=True, audio_output=True, graph=True, style="EAST", verbose=True)

Well recording after text "Recording, press any LETTER key to stop recording" appears, will create/replace output .wav file in synthesized piano notess, and will graph the data over time. Using style="EAST" if music runs on 5 notes oriental scale will greatly increase accuracy. Use style="ALL" if your music is complicated that doesn't run on a perticular scale. 


