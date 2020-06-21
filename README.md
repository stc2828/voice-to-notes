# Voice-to-notes
Taicheng Song's voice to notes project

This project take a .wav audio file of human singing, and output an optimal beats per minute, beats per note, frequency for of note, and duration of a beat(in ms).

The return format is a dictionary with: {"interval_length", "beats_per_note", "frequencies", "bpm"} as keys


### Important tip
Instead of sining the lyrics, try to sin with simple, burst of sound such as: "Ding, Ding, Ding", "Dong, Dong, Dong", etc. For better pickup accuracy.

This module find frequency with crepe, which on average take longer to run then the duration of the input sound file. 


### Example use case:
###### Case 1:
execute(record=True, audio_output=True, graph=True, style="EAST", verbose=True)

Will recording after text "Recording, press any LETTER key to stop recording" appears, then create/replace soundfile sheet_output.wav file with synthesized piano notes, and will graph the data over time. Using style="EAST" if music runs on 5 notes oriental scale will greatly increase accuracy. Use style="ALL" if your music is complicated that doesn't run on a particular scale. 
    
###### Case 2:
execute()

Will run every parameter on default as: 
execute(wav_filename="temps/output.wav", txt_filename="temps/output.wav", step_size=10, use_txt_input=False, record=False, audio_output=False, graph=False, style="WEST", scale=None, verbose=False, mute=False, piano=False)
It will execute on "temps/output.wav". If the file does not exist, it will print an error. 


### Other notes
There are 3 types of scales you can choose. 
style="WEST" is 7 notes standard scale
style="EAST" is 5 notes oriental scale
style="ALL" if your music is complicated that doesn't run on a particular scale.

If you know what specific scale you are using, valid scale values include:
'C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B'
Note that it is easy for human to sing off tune, it is advisable to leave scale=None and let module pick a best fit scale.

If you are using keyboard instead of human voice and obtain bad result, try piano=True. 

