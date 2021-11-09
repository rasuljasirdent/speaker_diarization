# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:00:09 2020

@author: rasul
"""

import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import math

#Constants that will be used to access timestamps 
START = 0
DURATION = 1
#Constants that will be used to link position in MFCC array with timestamps
WINSTEP = .01 #seconds between each the start of each MFCC frame
FRAMES_PER_SECOND = 1 / WINSTEP

""" get all wav files in a particular directory
    This function is taken directly from the gmm_example file"""
def get_audio_files(dir_path=None, extension='.wav'):
    wav_list = []

    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(extension):
                wav_list.append(os.path.join(dir_path,file))

    return wav_list
        
# Voice activity done using speech, music and noise model
# the result is a list of tuples
# each tuple contains:
# * label in 'Male', 'Female', 'Music', 'NOACTIVITY'
# * start time of the segment
# * end time of the segment
def detect_speech(seg, media):
    segmentation = seg(media)
    voices = set(["male", "female"])
    times = [segment for segment in segmentation if segment[0].lower() in voices]
    reformatted_times = []
    #Reformat the segments to just include start and duration in seconds
    for segment in times:
        start = float(segment[1])
        end = float(segment[2])
        duration = end - start
        new_record = tuple([start, duration])
        reformatted_times.append((new_record))
    return reformatted_times

"""This function takes the total list of MFCCs and gets rid of the non-speech
segments to create a new list of just speech MFCCs."""
def get_speech_mfccs(speech_times, total_mfcc, winstep=WINSTEP):
    #frames per second
    fps = 1 / winstep
    speech_mfcc = []
    for record in speech_times:
        #the times are in seconds and so we convert to frames
        start_frame = math.floor(record[START] * fps)
        duration = math.floor(record[DURATION] * fps)
        speech_mfcc.extend(total_mfcc[start_frame: start_frame + duration])
    return speech_mfcc

def assign_initial_segments(speech_mfcc, chunk_seconds=10, winstep=WINSTEP):
    speaker = 0
    initial_speakers = []
    current_frame = 0
    track_duration = len(speech_mfcc)
    fps = 1 / winstep
    target_duration = math.floor(chunk_seconds * fps)
    
    while current_frame < track_duration:
        frames_left = track_duration - current_frame
        #don't try to include nonexistent frames
        actual_duration = min(target_duration, frames_left)
        mfcc_chunk = speech_mfcc[current_frame: current_frame + actual_duration]
        initial_speakers.append([current_frame, mfcc_chunk, speaker])
        
        speaker += 1
        current_frame += actual_duration
    return initial_speakers
    
    
# get mfccs from a single wav file
def extract_mfcc_from_wav_file(file_path=None):
    mfccs = []

    if os.path.exists(file_path):
        (fs, sig) = wav.read(file_path)
        mfccs = mfcc(signal=sig,
                     samplerate=fs,
                     winlen=0.025,
                     winstep=WINSTEP,
                     numcep=13,
                     nfilt=26,
                     nfft=512)

    return mfccs

def extract_speech_mfccs_wav(segmenter, file_path=None):
    if os.path.exists(file_path):
        file_mfcc = extract_mfcc_from_wav_file(file_path=file_path)
        speech = detect_speech(segmenter, file_path)
        return get_speech_mfccs(speech, file_mfcc)
    

""" collect all speech mfccs from a directory (non-recursive)
    I have adapted the file from the gmm example to include voice activity
    detection as part of the extraction process so that the ubm will have
    less noise"""
def get_mfccs_in_directory(segmenter, dir_path=None):
    # get list of files
    files = get_audio_files(dir_path=dir_path)
    directory_mfcc = []
    for file in files:
        file_mfcc = extract_speech_mfccs_wav(segmenter, file)
        directory_mfcc.extend(file_mfcc)
    return directory_mfcc