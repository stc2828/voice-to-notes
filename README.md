# **Voice-to-notes**
This is Taicheng Song's voice to notes project.

This module take a .wav audio file of human singing, and output the optimal beats per minute, beats per note, frequency for of note, and duration of a beat(in ms).

The return format is a dictionary with: {"interval_length", "beats_per_note", "frequencies", "bpm"} as keys.


## **Important tips**
For better pickup accuracy, instead of singing the lyrics, try to sing with simple, burst of sound, like: "Ding, Ding, Ding", "Dong, Dong, Dong", etc. 

Please sing at moderate(slow) pace.

Try not to input more than 20 seconds at a time. 

This module find frequency with crepe, which on average take longer to run than the duration of the input sound file.

You must install crepe, which only works on Python 3. (Installation help below)

you must install pyaudio if you want the module to record your voice.

The bpm could be high since the algorithm tend to look for the shortest reasonable beat size. 


## **Example use cases**
#### *Case 1:*
`execute(record=True, audio_output=True, graph=True, style="EAST", verbose=True)`

Will record after text "Recording, press any LETTER key to stop recording" appear, then create/replace soundfile sheet_output.wav file with synthesized piano notes, and will graph the data over time. Using style="EAST" if the music runs on 5 notes oriental scale will greatly increase accuracy. Use style="ALL" if the music is complicated, that doesn't run on a particular scale. 
    
#### *Case 2:*
`execute()`

Will run every parameter on default as: 
execute(wav_filename="temps/output.wav", txt_filename="temps/output.wav", step_size=10, use_txt_input=False, record=False, audio_output=False, graph=False, style="WEST", scale=None, verbose=False, mute=False, piano=False)
In this case, it will look for "temps/output.wav". If the file does not exist, it will print an error. 


## **Other notes**
There are 3 types of scales you can choose:
* `style="WEST"` is 7 notes standard scale.
* `style="EAST"` is 5 notes oriental scale.
* `style="ALL"` is used when your music is so complicated that doesn't run on a particular scale.

`scale=None` is set at defult since it is easy for human to sing off tune, and it is advisable to let the module pick a best fit scale. However, if you know what specific scale you are using, valid scale values are:
'C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B'

If you are using keyboard/piano, instead of human voice, and obtained bad result, try `piano=True`. 

#### *Quick installation guide:*
Make sure you have python 3 installed.

`python get-pip.py`

`pip install crepe`

`pip install PyAudio`
