import os
import pyaudio
import wave
import crepe
import time
from scipy.io import wavfile
from pynput import keyboard
import numpy as np


# GLOBALS
keep_recording = True
VERBOSE = False
MUTE = False
ratios = [1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1, 3 / 2, 2, 3, 4, 5, 6, 7, 8]
list_of_notes = [32.7,34.65,36.71,38.89,41.2,43.65,46.25,49,51.91,55,58.27,61.74,65.41,69.3,73.42,77.78,82.41,87.31,92.5,98,103.83,110,116.54,123.47,130.81,138.59,146.83,155.56,164.81,174.61,185,196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.3,440,466.16,493.88,523.25,554.37,587.33,622.25,659.25,698.46,739.99,783.99,830.61,880,932.33,987.77,1046.5,1108.73,1174.66,1244.51,1318.51,1396.91,1479.98,1567.98,1661.22,1760,1864.66,1975.53,2093]
west_scales = [[32.7, 36.71, 41.2, 43.65, 49, 55, 61.74, 65.41, 73.42, 82.41, 87.31, 98, 110, 123.47, 130.81, 146.83, 164.81, 174.61, 196, 220, 246.94, 261.63, 293.66, 329.63, 349.23, 392, 440, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880, 987.77, 1046.5, 1174.66, 1318.51, 1396.91, 1567.98, 1760, 1975.53, 2093], [34.65, 38.89, 43.65, 46.25, 51.91, 58.27, 65.41, 69.3, 77.78, 87.31, 92.5, 103.83, 116.54, 130.81, 138.59, 155.56, 174.61, 185, 207.65, 233.08, 261.63, 277.18, 311.13, 349.23, 369.99, 415.3, 466.16, 523.25, 554.37, 622.25, 698.46, 739.99, 830.61, 932.33, 1046.5, 1108.73, 1244.51, 1396.91, 1479.98, 1661.22, 1864.66, 2093], [36.71, 41.2, 46.25, 49, 55, 61.74, 69.3, 73.42, 82.41, 92.5, 98, 110, 123.47, 138.59, 146.83, 164.81, 185, 196, 220, 246.94, 277.18, 293.66, 329.63, 369.99, 392, 440, 493.88, 554.37, 587.33, 659.25, 739.99, 783.99, 880, 987.77, 1108.73, 1174.66, 1318.51, 1479.98, 1567.98, 1760, 1975.53], [38.89, 43.65, 49, 51.91, 58.27, 65.41, 73.42, 77.78, 87.31, 98, 103.83, 116.54, 130.81, 146.83, 155.56, 174.61, 196, 207.65, 233.08, 261.63, 293.66, 311.13, 349.23, 392, 415.3, 466.16, 523.25, 587.33, 622.25, 698.46, 783.99, 830.61, 932.33, 1046.5, 1174.66, 1244.51, 1396.91, 1567.98, 1661.22, 1864.66, 2093], [41.2, 46.25, 51.91, 55, 61.74, 69.3, 77.78, 82.41, 92.5, 103.83, 110, 123.47, 138.59, 155.56, 164.81, 185, 207.65, 220, 246.94, 277.18, 311.13, 329.63, 369.99, 415.3, 440, 493.88, 554.37, 622.25, 659.25, 739.99, 830.61, 880, 987.77, 1108.73, 1244.51, 1318.51, 1479.98, 1661.22, 1760, 1975.53], [43.65, 49, 55, 58.27, 65.41, 73.42, 82.41, 87.31, 98, 110, 116.54, 130.81, 146.83, 164.81, 174.61, 196, 220, 233.08, 261.63, 293.66, 329.63, 349.23, 392, 440, 466.16, 523.25, 587.33, 659.25, 698.46, 783.99, 880, 932.33, 1046.5, 1174.66, 1318.51, 1396.91, 1567.98, 1760, 1864.66, 2093], [46.25, 51.91, 58.27, 61.74, 69.3, 77.78, 87.31, 92.5, 103.83, 116.54, 123.47, 138.59, 155.56, 174.61, 185, 207.65, 233.08, 246.94, 277.18, 311.13, 349.23, 369.99, 415.3, 466.16, 493.88, 554.37, 622.25, 698.46, 739.99, 830.61, 932.33, 987.77, 1108.73, 1244.51, 1396.91, 1479.98, 1661.22, 1864.66, 1975.53], [49, 55, 61.74, 65.41, 73.42, 82.41, 92.5, 98, 110, 123.47, 130.81, 146.83, 164.81, 185, 196, 220, 246.94, 261.63, 293.66, 329.63, 369.99, 392, 440, 493.88, 523.25, 587.33, 659.25, 739.99, 783.99, 880, 987.77, 1046.5, 1174.66, 1318.51, 1479.98, 1567.98, 1760, 1975.53, 2093], [51.91, 58.27, 65.41, 69.3, 77.78, 87.31, 98, 103.83, 116.54, 130.81, 138.59, 155.56, 174.61, 196, 207.65, 233.08, 261.63, 277.18, 311.13, 349.23, 392, 415.3, 466.16, 523.25, 554.37, 622.25, 698.46, 783.99, 830.61, 932.33, 1046.5, 1108.73, 1244.51, 1396.91, 1567.98, 1661.22, 1864.66, 2093], [55, 61.74, 69.3, 73.42, 82.41, 92.5, 103.83, 110, 123.47, 138.59, 146.83, 164.81, 185, 207.65, 220, 246.94, 277.18, 293.66, 329.63, 369.99, 415.3, 440, 493.88, 554.37, 587.33, 659.25, 739.99, 830.61, 880, 987.77, 1108.73, 1174.66, 1318.51, 1479.98, 1661.22, 1760, 1975.53], [58.27, 65.41, 73.42, 77.78, 87.31, 98, 110, 116.54, 130.81, 146.83, 155.56, 174.61, 196, 220, 233.08, 261.63, 293.66, 311.13, 349.23, 392, 440, 466.16, 523.25, 587.33, 622.25, 698.46, 783.99, 880, 932.33, 1046.5, 1174.66, 1244.51, 1396.91, 1567.98, 1760, 1864.66, 2093], [61.74, 69.3, 77.78, 82.41, 92.5, 103.83, 116.54, 123.47, 138.59, 155.56, 164.81, 185, 207.65, 233.08, 246.94, 277.18, 311.13, 329.63, 369.99, 415.3, 466.16, 493.88, 554.37, 622.25, 659.25, 739.99, 830.61, 932.33, 987.77, 1108.73, 1244.51, 1318.51, 1479.98, 1661.22, 1864.66, 1975.53]]
east_scales = [[32.7, 38.89, 43.65, 51.91, 58.27, 65.41, 77.78, 87.31, 103.83, 116.54, 130.81, 155.56, 174.61, 207.65, 233.08, 261.63, 311.13, 349.23, 415.3, 466.16, 523.25, 622.25, 698.46, 830.61, 932.33, 1046.5, 1244.51, 1396.91, 1661.22, 1864.66, 2093], [34.65, 41.2, 46.25, 55, 61.74, 69.3, 82.41, 92.5, 110, 123.47, 138.59, 164.81, 185, 220, 246.94, 277.18, 329.63, 369.99, 440, 493.88, 554.37, 659.25, 739.99, 880, 987.77, 1108.73, 1318.51, 1479.98, 1760, 1975.53], [36.71, 43.65, 49, 58.27, 65.41, 73.42, 87.31, 98, 116.54, 130.81, 146.83, 174.61, 196, 233.08, 261.63, 293.66, 349.23, 392, 466.16, 523.25, 587.33, 698.46, 783.99, 932.33, 1046.5, 1174.66, 1396.91, 1567.98, 1864.66, 2093], [38.89, 46.25, 51.91, 61.74, 69.3, 77.78, 92.5, 103.83, 123.47, 138.59, 155.56, 185, 207.65, 246.94, 277.18, 311.13, 369.99, 415.3, 493.88, 554.37, 622.25, 739.99, 830.61, 987.77, 1108.73, 1244.51, 1479.98, 1661.22, 1975.53], [41.2, 49, 55, 65.41, 73.42, 82.41, 98, 110, 130.81, 146.83, 164.81, 196, 220, 261.63, 293.66, 329.63, 392, 440, 523.25, 587.33, 659.25, 783.99, 880, 1046.5, 1174.66, 1318.51, 1567.98, 1760, 2093], [43.65, 51.91, 58.27, 69.3, 77.78, 87.31, 103.83, 116.54, 138.59, 155.56, 174.61, 207.65, 233.08, 277.18, 311.13, 349.23, 415.3, 466.16, 554.37, 622.25, 698.46, 830.61, 932.33, 1108.73, 1244.51, 1396.91, 1661.22, 1864.66], [46.25, 55, 61.74, 73.42, 82.41, 92.5, 110, 123.47, 146.83, 164.81, 185, 220, 246.94, 293.66, 329.63, 369.99, 440, 493.88, 587.33, 659.25, 739.99, 880, 987.77, 1174.66, 1318.51, 1479.98, 1760, 1975.53], [49, 58.27, 65.41, 77.78, 87.31, 98, 116.54, 130.81, 155.56, 174.61, 196, 233.08, 261.63, 311.13, 349.23, 392, 466.16, 523.25, 622.25, 698.46, 783.99, 932.33, 1046.5, 1244.51, 1396.91, 1567.98, 1864.66, 2093], [51.91, 61.74, 69.3, 82.41, 92.5, 103.83, 123.47, 138.59, 164.81, 185, 207.65, 246.94, 277.18, 329.63, 369.99, 415.3, 493.88, 554.37, 659.25, 739.99, 830.61, 987.77, 1108.73, 1318.51, 1479.98, 1661.22, 1975.53], [55, 65.41, 73.42, 87.31, 98, 110, 130.81, 146.83, 174.61, 196, 220, 261.63, 293.66, 349.23, 392, 440, 523.25, 587.33, 698.46, 783.99, 880, 1046.5, 1174.66, 1396.91, 1567.98, 1760, 2093], [58.27, 69.3, 77.78, 92.5, 103.83, 116.54, 138.59, 155.56, 185, 207.65, 233.08, 277.18, 311.13, 369.99, 415.3, 466.16, 554.37, 622.25, 739.99, 830.61, 932.33, 1108.73, 1244.51, 1479.98, 1661.22, 1864.66], [61.74, 73.42, 82.41, 98, 110, 123.47, 146.83, 164.81, 196, 220, 246.94, 293.66, 329.63, 392, 440, 493.88, 587.33, 659.25, 783.99, 880, 987.77, 1174.66, 1318.51, 1567.98, 1760, 1975.53]]


# Functions
def generate_necessary_directories():
    if not os.path.exists("temps"):
        os.mkdir("temps")


def record_audio(output_directory="temps/output.wav"):
    def on_press(key):
        global keep_recording, VERBOSE
        keep_recording = False
        if VERBOSE:
            print("stopping")
        return False
    def on_release(key):
        pass

    global keep_recording, VERBOSE, MUTE
    keep_recording = True

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    timeout = 100  # max length of recording in seconds
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    # keyboard controller
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    if not MUTE:
        print('Recording, press any key to stop recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []  # Initialize array to store frames
    while keep_recording and timeout > 0:
        timeout -= chunk / fs
        data = stream.read(chunk)
        frames.append(data)

    # cut out a bit in front and back to avoid the mic burst, 11 is about 1/4 sec
    frames = frames[2:-1]
    listener.join()

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    if VERBOSE:
        print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(output_directory, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return output_directory


def amplitude(filename, time_step):
    fs, data = wavfile.read(filename)
    #data = data[:200000]
    step_size = fs*time_step/1000
    step_index = 1
    arr = []
    temp_sum = 0
    temp_len = 0

    try:
        if len(data[0]) >= 1:
            for i, val in enumerate(data):
                if i > step_index * step_size:
                    step_index += 1
                    if temp_len > 0:
                        arr.append(temp_sum / temp_len)
                    temp_sum = 0
                    temp_len = 0
                # continue anyway
                # tmp = val[0]
                temp_sum += abs(val[0])
                temp_len += 1
            if temp_len > 0:
                arr.append(temp_sum / temp_len)
    except:
        for i, val in enumerate(data):
            if i > step_index * step_size:
                step_index += 1
                if temp_len>0:
                    arr.append(temp_sum / temp_len)
                temp_sum = 0
                temp_len = 0
            #continue anyway
            temp_sum += abs(val)
            temp_len += 1
        if temp_len > 0:
            arr.append(temp_sum / temp_len)

    return arr


def run_crepe(filename, step_size, smaller_model=True):
    sr, audio = wavfile.read(filename)
    if smaller_model:
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=step_size, verbose=False, model_capacity="medium")
    else:
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=step_size, verbose=False)
    return time, frequency, confidence


def record_to_txt(amp, fre, con, filename="temps/output.txt"):
    f = open(filename, "w")
    for i in range(len(amp)):
        f.write(str(amp[i]) + "," + str(fre[i]) + "," + str(con[i]) + "\n")
    f.close()


def smooth_amplitude(arr):
    # make every value the average of adjacent 5
    l = len(arr)

    s = sum(arr[:5])
    new_arr = [sum(arr[:3]) / 3, sum(arr[:4]) / 4, s / 5]
    for i in range(3, l-2):
        s += arr[i + 2] - arr[i-3]
        new_arr.append(s / 5)
    new_arr.append(sum(arr[-4:]) / 4)
    new_arr.append(sum(arr[-3:]) / 3)

    return new_arr


def smooth_frequency(frequency, confidence):
    # make every value the average of adjacent 5, weighted by confidence
    l = len(frequency)

    f = sum([frequency[i]*confidence[i] for i in range(5)])
    w = sum([confidence[i] for i in range(5)])
    new_arr = [f/w, f/w, f/w]
    for i in range(3, l-2):
        f += frequency[i + 2]*confidence[i + 2] - frequency[i - 3]*confidence[i - 3]
        w += confidence[i + 2] - confidence[i - 3]
        new_arr.append(f / w)
    new_arr.append(f / w)
    new_arr.append(f / w)

    return new_arr


def find_beat(amp, fre, con, min_len=7, piano=False):
    # min_len amount of steps for sound to be a beat and not a fluke
    def rate(amplitude, confidence, std_amplitude):
        if confidence > 0.7:
            return amplitude + (confidence - 0.7) * 2 * std_amplitude
        if confidence < 0.7:
            return amplitude - (0.7 - confidence) * 2 * std_amplitude
        return amplitude

    if piano: # piano input cause significant offset for some reason
        amp = [0]*3 + amp[:-3]
    q9_amp = np.quantile(amp, 0.9)
    min_amp = q9_amp*0.25
    i, l = 0, len(amp)
    arr = []

    while i < l:
        if rate(amp[i], con[i], q9_amp) >= min_amp:
            in_beat = True

            # sound must last min_len to be considered
            # weight multiplier increase over time to give later notes more weight
            start = i
            beat_len = 0
            sum_fre = 0
            sum_rat = 0
            sum_rat_weight = 0
            sum_fre_weight = 0
            sum_fre_unweighted = 0
            while i < start + min_len and i < l:
                cur_val = rate(amp[i], con[i], q9_amp)
                if cur_val < min_amp:
                        in_beat = False
                        # beat not recorded, alg continue on
                        break

                beat_len += 1
                sum_fre_unweighted += fre[i]
                sum_fre += fre[i] * con[i]*beat_len
                sum_fre_weight += con[i] * beat_len
                sum_rat += cur_val * (beat_len**2)
                sum_rat_weight += beat_len**2
                i += 1

            # if the beat will be recorded, find its end
            if in_beat:
                while i < l and rate(amp[i], con[i], q9_amp) > min_amp:
                    cur_val = rate(amp[i], con[i], q9_amp)
                    avg_rat = sum_rat / sum_rat_weight
                    avg_fre = sum_fre / sum_fre_weight

                    # if there is a big drop in rating, wrap up this beat, but push i until inflection
                    if cur_val < avg_rat*0.65:
                        break

                    # if there is a frequency difference that last for min_len*x, cut off
                    # give triple min_len to stabilize avg_freq
                    if i > int(start + min_len*1.6) and abs(avg_fre - fre[i]) / avg_fre > 0.075:
                        sum_temp_fre = 0
                        temp_i = i
                        temp_weight = 0
                        while temp_i < i + min_len and temp_i < l:
                            sum_temp_fre += fre[temp_i] * con[temp_i]
                            temp_weight += con[temp_i]
                            temp_i += 1
                        avg_temp_fre = sum_temp_fre / temp_weight

                        if abs(avg_fre - avg_temp_fre) / avg_fre > 0.075:
                            break

                    # update variables
                    beat_len += 1
                    sum_fre_unweighted += fre[i]
                    sum_fre += fre[i] * con[i]*beat_len
                    sum_rat += cur_val * (beat_len**2)
                    sum_fre_weight += con[i]*beat_len
                    sum_rat_weight += (beat_len**2)
                    i += 1

                # record beat as: [start, end, frequency]
                arr.append([start, i, sum_fre_unweighted / beat_len])

                # leave i at an inflection point
                prev_amp = rate(amp[i-1], con[i-1], q9_amp)
                while i<l and rate(amp[i], con[i], q9_amp) <= prev_amp:
                    prev_amp = rate(amp[i], con[i], q9_amp)
                    i += 1
        i += 1

    return arr


# input list of notes with their starting location
def seek(notes):
    global ratios
    def find_nearest(n):

        if n < ratios[0] * 0.8:
            return -2, None
        if n > ratios[-1] * 1.2:
            return -1, None

        idx = -1
        min_dist = 10000
        for i, val in enumerate(ratios):
            if abs(val - n) < min_dist:
                min_dist = abs(val - n)
                idx = i
        return idx, min_dist

    start = notes[0]
    durations = []
    for i in notes[1:]:
        durations.append(i-start)
        start = i
    l = len(durations)

    #scoring system where int: 1pt, 1/2 for 0.4pt, other_fraction: 0pt, not_in_ratios: -4pt
    scores = []

    #find good_interval base on score
    #copy a list of durations with +-5% virations
    durations2 = []
    for i in durations:
        durations2 += [i*0.9] + [i*0.95] + [i] + [i*1.05] + [i*1.1]
    temp_ratios = ratios + [ratios[0]/2] + [ratios[-1]*2]
    for i, temp_interval in enumerate(durations2):
        score = 0
        for j, temp_duration in enumerate(durations):
            idx, err = find_nearest(temp_duration/temp_interval)
            temp = temp_ratios[idx]

            #scoring base on number of ints, 0.5 get partial credit
            if idx < 0:
                score -= 4
            elif isinstance(temp, int):
                score += 1
            elif temp == 0.5:
                score += 0.4
        scores.append(score)
    max_score = max(scores)
    i = 0
    while scores[i] != max_score:
        i += 1
    good_interval = durations2[i]

    #find best_interval within +-5% spread with score
    grid = [[0 for _ in range(l)] for _ in range(10)]
    scores = []
    intervals = [good_interval*(0.9+0.02*i) for i in range(10)]
    for i, temp_interval in enumerate(intervals):
        score = 0
        for j, temp_duration in enumerate(durations):
            idx, err = find_nearest(temp_duration/temp_interval)
            temp = temp_ratios[idx]
            grid[i][j] = temp

            #scoring base on number of ints
            if idx < 0:
                score -= 4
            elif temp == int(temp):
                score += 1
            elif temp == 0.5:
                score += 0.4
        scores.append(score)
    max_score = max(scores)
    i = 0
    while scores[i] != max_score:
        i += 1
    best_interval = intervals[i]

    return grid[i], best_interval


def improve(intervals):
    global ratios
    def adj_ratio(x):
        # return smaller, larger ratio.
        ratios2 = [None] + ratios + [None]

        for i in range(len(ratios)):
            if abs(ratios[i] - x) < 0.001:
                return ratios2[i], ratios2[i + 2]
        return None, None

    l = len(intervals)
    shift_cost = 0.6

    mem = {} # memory format: (index, offset) : score, temp_interval
    # offsets are rounded to 4 after decimal

    def dfs(index, offset):
        if index == l:
            return 0, []

        val = intervals[index]
        if abs((offset + val) % 1) <= 0.001:
            if (index+1, 0) in mem:
                score, temp_interval = mem[index + 1, 0]
            else:
                score, temp_interval = dfs(index + 1, 0)

            mem[index, 0] = score + 1, [val] + temp_interval
            return mem[index, 0]

        else:
            # making changes will -0.5 score, no changes do not change score
            ss, ll = adj_ratio(val)

            s_score, S = -10, []
            if ss is not None:
                if (index + 1, round((offset + ss)%1, 4)) in mem:
                    s_score, S = mem[index + 1, round((offset + ss)%1, 4)]
                else:
                    s_score, S = dfs(index + 1, offset + ss)

            if (index + 1, round((offset + val) % 1, 4)) in mem:
                m_score, M = mem[index + 1, round((offset + val) % 1, 4)]
            else:
                m_score, M = dfs(index + 1, offset + val)

            l_score, L = -10, []
            if ll is not None:
                if (index + 1, round((offset + ll) % 1, 4)) in mem:
                    l_score, L = mem[index + 1, round((offset + ll) % 1, 4)]
                else:
                    l_score, L = dfs(index + 1, offset + ll)

            ms = max([s_score-0.5,m_score,l_score-0.5])
            if m_score == ms:
                mem[index, round(offset % 1, 4)] = m_score, [val] + M
                return m_score, [val] + M
            if s_score-0.5 == ms:
                mem[index, round(offset % 1, 4)] = s_score-0.5, [ss] + S
                return s_score-0.5, [ss] + S
            if l_score-0.5 == ms:
                mem[index, round(offset % 1, 4)] = l_score-0.5, [ll] + L
                return l_score-0.5, [ll] + L

    ret_score, arr = dfs(0, 0)
    tail = 4-(sum(arr))%4
    if tail<1:
        tail += 4
    return ret_score, arr + [tail]


def find_note(frequencies, scale_index=None, style='WEST'):
    # take a list of frequencies, output notes base on optimal scale, and the scale used
    def match_to_scale(freq, scale):
        # return min_gap as 1 base percentage
        if freq < scale[0]:
            return 0, (scale[0] - freq) / freq

        i = 1
        l = len(scale)
        while i < l:
            if freq < scale[i]:
                # front back could add up to be less than 1
                front, back = freq - scale[i - 1], (scale[i] - freq) * 0.98
                if front < back:
                    return i - 1, front / freq
                else:
                    return i, back / freq
            i += 1

        return l, (freq - scale[-1]) / freq

    def match(freqs, scale):
        # match frequencies to a scale, return delta_score and notes
        score = 0
        notes = []
        for f in freqs:
            note_index, temp_score = match_to_scale(f, scale)
            notes.append(scale[note_index])
            score += temp_score

        return score, notes

    global list_of_notes, west_scales, east_scales
    if style not in ["EAST", "WEST"]:
        _, sheet = match(frequencies, list_of_notes)
        return sheet, scale_index
    if scale_index == None:
        best_score = 9999  # serve as int_max
        best_sheet = []
        best_scale = -1
        if style == 'WEST':
            scales = west_scales
        else:
            scales = east_scales

        sheets = []
        for i in range(len(scales)):
            temp_score, sheet = match(frequencies, scales[i])
            if temp_score < best_score:
                best_score = temp_score
                best_sheet = sheet
                best_scale = i
            sheets.append(sheet)
        if best_sheet == sheets[0]:
            best_scale = 0

        return best_sheet, best_scale

    else:
        if scale_index == -1:
            scale_to_use = list_of_notes
        elif 0 <= scale_index <= 11:
            if style == 'WEST':
                scale_to_use = west_scales[scale_index]
            else:
                scale_to_use = east_scales[scale_index]
        else:
            print("unknown scale_index")
            return None, None
        _, sheet = match(frequencies, scale_to_use)
        return sheet, scale_index


def import_reference_soundtrack(filename, note_length, number_notes):
    fs, data = wavfile.read(filename)
    ret = []
    for i in range(number_notes):
        ret.append([i for i, _ in data[i * note_length:(i + 1) * note_length - 1]])
    return ret


def interval_to_sheet(intervals, frequencies, interval_size):
    # interval_size in seconds
    sheet = []
    t = 1
    for i, units_of_intervals in enumerate(intervals):
        # rounding to 4 decimals
        sheet.append([frequencies[i], 1, t, round(t + 0.75 * units_of_intervals * interval_size, 4)])
        t += units_of_intervals * interval_size
        t = round(t, 4)
    return sheet


def sheet_to_audio(sheet, reference, fs=44100):
    global list_of_notes, VERBOSE

    def frequency_to_reference(frequency):
        for i, val in enumerate(list_of_notes):
            if abs(val - frequency) <= 0.5:
                return i
        return -1

    filename = "sheet_output.wav"
    if VERBOSE:
        print("Preparing audio output")
    # format: [[note_frequency, volume, starting, ending]], sheet_frequency

    # add 0.1 sec padding front and back
    length_of_music = int(max([e for _, _, _, e in sheet]) * fs) + 4410 + 4410
    arr = [0] * length_of_music
    front_padding = 4410

    for f, v, s, e in sheet:
        s, e = int(s * fs) + front_padding, int(e * fs) + front_padding

        # pickup 50 early, tone down 50 early
        sheet_index = s - 50
        note_index = 0
        note = reference[frequency_to_reference(f)]
        note_size = len(note)
        while sheet_index < e:
            arr[sheet_index] += note[note_index]
            sheet_index += 1
            note_index += 1
            if note_index >= note_size:
                note_index = int(note_size / 4 * 3)

        # take large drop then 200 tick to end a note gradually
        while sheet_index < e + 200:
            arr[sheet_index] += note[note_index] * ((e - sheet_index) / 100)
            sheet_index += 1
            note_index += 1
            if note_index >= note_size:
                note_index = int(note_size / 4 * 3)

    out = np.zeros(shape=(length_of_music, 2), dtype=np.int16)
    for i, val in enumerate(arr):
        out[i] += int(val)
    wavfile.write(filename, 44100, out)
    if VERBOSE:
        print("Audio output completed")
    return filename


### MAIN
def execute(wav_filename="temps/output.wav", txt_filename="temps/output.txt", step_size=10, use_txt_input=False, record=False, audio_output=False, graph=False, style="WEST", scale=None, verbose=False, mute=False, express=True, piano=False):
    global VERBOSE, MUTE
    generate_necessary_directories()
    if verbose:
        VERBOSE = True
    if mute:
        MUTE = True
    if use_txt_input:
        amp, fre, con = [], [], []
        f = open(txt_filename, "r")
        data = f.read()
        for i in data.split():
            temp = i.split(",")
            amp.append(float(temp[0]))
            fre.append(float(temp[1]))
            con.append(float(temp[2]))
        f.close()
    else:
        if record:
            wav_filename = record_audio()

        try:
            amp = amplitude(wav_filename, step_size)
        except:
            print("WAV file: '", wav_filename, "' not found")
            return {}
        start_time = time.time()
        tim, fre, con = run_crepe(wav_filename, step_size, smaller_model=express)
        record_to_txt(amp, fre, con, filename=txt_filename)
        if VERBOSE:
            print("crepe runtime: ", int(1000*(time.time()-start_time)), "ms")

    # error handling for "scale"
    try:
        if 0<=scale<=11:
            list_of_scales = [['C'], ['C+', 'D-'], ['D'], ['D+', 'E-'], ['E'], ['F'], ['F+', 'G-'], ['G'], ['G+', 'A-'], ['A'], ['A+', 'B-'], ['B']]
            i = 0
            while i <= 11:
                if scale in list_of_scales[i]:
                    scale = i
                    break
                i += 1
            if i == 12:
                scale = None
    except:
        scale = None

    con = smooth_amplitude(con)
    fre = smooth_frequency(fre, con)
    amp = smooth_amplitude(amp)

    beats = find_beat(amp, fre, con, min_len=7, piano=piano)
    start_of_beats = [i[0] for i in beats]

    intervals, interval_size = seek(start_of_beats)
    if VERBOSE:
        print("Beat duration: ", interval_size*step_size,  "ms")
        print("Raw beats per note: ", intervals)
        print("Raw frequencies: ", [i for _, _, i in beats])

    score, intervals = improve(intervals)
    freqs, scale_index = find_note([i for _,_,i in beats], scale_index=scale, style=style)
    if scale_index == None:
        scale_index = -1
    if VERBOSE:
        scales = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B', 'C, East', 'C+, East', 'D, East', 'D+, East', 'E, East', 'F, East', 'F+, East', 'G, East', 'G+, East', 'A, East', 'A+, East', 'B, East', 'ALL']
        print("Scale used: ", scales[scale_index])

    if audio_output:
        sheet = interval_to_sheet(intervals, freqs, interval_size * (step_size * 0.001))
        ref = import_reference_soundtrack('audio_output_refrence.wav', 44100 * 2, 73)
        sheet_path = sheet_to_audio(sheet, ref)
        if VERBOSE:
            print("Sound output saved at: ", sheet_path)
            print("Music sheet: ")
            print("units_of_intervals, [frequency, strength, start(sec), end(sec)]")
            for i in range(len(beats)):
                print(intervals[i], sheet[i])

    if graph:
        import matplotlib.pylab as plt
        bins = range(len(amp))
        if piano:
            amp = [0] * 3 + amp[:-3]
        plt.plot(bins, amp, label="amplitude", color="red")
        plt.plot(bins, [10 * i for i in fre], label="frequency(Hz)(x10)", color="blue")
        plt.plot(bins, [1000 * i for i in con], label="confidence(%)(x10)", color="green")
        plt.axvline(x=start_of_beats[0], alpha=0.5, label="start of each beat", color="grey", linestyle="dashed")
        if len(start_of_beats) > 1:
            for i in start_of_beats[1:]:
                plt.axvline(x=i, alpha=0.5, color="grey", linestyle="dashed")
        plt.xlabel("Time(ms)")
        plt.title("amplitude, frequency, confidence over time")
        plt.grid(alpha=0.25)
        plt.legend(loc='upper right')
        plt.show()

    ret = {"beat_size":interval_size, "beats":intervals, "frequencies":freqs, "bpm":int(60000/interval_size/step_size)}
    return ret


if __name__ == "__main__":
    ret = execute(record=True, verbose=True, audio_output=True)
    print(ret)
