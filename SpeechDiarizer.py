import os
import copy
from sklearn import mixture
from collections import defaultdict, Counter
import pickle
import extract
from inaSpeechSegmenter import Segmenter
import numpy as np
import timeit


# build a gmm with sklearn
# covariance types are 'spherical', 'tied', 'diag', 'full'
#function supplied by instructor
def build_gmm(data=None, components =8):
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=components,
                                  covariance_type='diag',
                                  max_iter=100,
                                  warm_start=True,
                                  init_params="kmeans")
    gmm.fit(data)
    return gmm

#function supplied by instructor
def adapt_fit(ubm_model, data):
    adapted_gmm = copy.deepcopy(ubm_model)
    adapted_gmm.fit(data)
    return adapted_gmm

#function supplied by instructor
def build_speech_ubm(data_directory, pkl_filename, segmenter):
    global_mfcc_data = extract.get_mfccs_in_directory(segmenter,
        dir_path=data_directory)
    ubm_model = build_gmm(data = global_mfcc_data, components=8)
    # Save GMM to disk
    with open(pkl_filename, 'wb') as file:
        pickle.dump(ubm_model, file)
    return ubm_model


# Load GMM from disk and re-check weights
#function supplied by instructor
def load_gmm(pkl_filename):
    print("Re-loaded weights")
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    print(pickle_model.weights_)
    return pickle_model


WINSTEP = extract.WINSTEP
START_FRAME = 0
MODEL = 0
MFCC = 1
LABEL = 2
MERGE_SCORE = 2

KEEP = 0
DELETE = 1
MERGE_TUPLE = 0
ORDER = 1
MATRIX = 2

"""This creates a dictionary where each key is the label for a speaker and the values
are lists containing the gmm in the 0 position and the MFCC used to generate it 
in the 1 position"""

def create_speaker_gmm(labelled_mfcc, ubm):
    speaker_gmm = defaultdict(lambda: [])
    #gather up all the data associated with a speaker
    for segment in labelled_mfcc:
        speaker = segment[LABEL]
        speaker_gmm[speaker].extend((segment[MFCC]))
    #convert the mfccs into gmms
    for speaker in speaker_gmm:
        data = speaker_gmm[speaker]
        my_gmm = build_gmm(data)
        #keeping the mfcc will make merging faster later
        #we are inserting to make the MFCC the second element of the list for consistency
        #so we need to turn our list into a list of lists
        speaker_gmm[speaker] = [my_gmm, data]
    return speaker_gmm

"""This functions asks each individual MFCC for vote for the GMM most likely to produce it.
After all of the MFCC for a segment have voted, the segment is relabel with whichever GMM receives the
most votes."""
def relabel_mfcc(labelled_mfcc, speaker_gmm):
    for segment in labelled_mfcc:
        records = segment[MFCC]
        most_likely = list([0 for i, r in enumerate(records)])
        best_likelihood = list([-1000 for i, r in enumerate(records)])
        #assign a most likely gmm for each individual frame
        for speaker in speaker_gmm:
            likelihoods = speaker_gmm[speaker][MODEL].score_samples(segment[MFCC])
            for record, chance in enumerate(likelihoods):
                if chance >= best_likelihood[record]:
                    most_likely[record] = speaker
                    best_likelihood[record] = chance
        # take the most frequently assigned as the label for the segment
        most_popular = Counter(most_likely).most_common(1)[0]
        segment[LABEL] = most_popular[0]
    return  labelled_mfcc

"""This function creates a merged GMM using the data of 2 input GMM. The combined
model components number 3/4 * the sum of input models' components."""
def try_merge(gmms, speaker1, speaker2):
    mod1, mod2 = gmms[speaker1][MODEL], gmms[speaker2][MODEL]
    mfcc1, mfcc2  = list(gmms[speaker1][MFCC]), list(gmms[speaker2][MFCC])
    comp1, comp2 = len(mod1.means_), len(mod2.means_)
    merge_data = mfcc1
    merge_data.extend(mfcc2)
    merge_data = np.array(merge_data)
    #add more components but do not double
    merge_gmm = build_gmm(merge_data, components= round((comp1 + comp2) * 3 /4))
    return [merge_gmm, merge_data]

"""This function calculates the bic scores of two separate gmms and returns
the improvement in bic that would come from their merger"""
def find_improvement(gmms, speaker1, speaker2, merged):
    mfcc1, mfcc2  = np.array(gmms[speaker1][MFCC]), np.array(gmms[speaker2][MFCC])
    combined_bic = merged[MODEL].bic(merged[MFCC])
    bic1 = gmms[speaker1][MODEL].bic(mfcc1)
    bic2 = gmms[speaker2][MODEL].bic(mfcc2)
    return -(combined_bic - (bic1 + bic2))
"""This function initializes the scoring matrix by calculating the bic improvement 
of each possible merger. Because this is a reciprocal matrix, it is only necessary
to fill out one triangle."""
def fill_score_matrix(gmms, ordered_speakers, score_matrix):
    for pos1, speaker1 in enumerate(ordered_speakers):
        #we need a new second iterable each time
        for pos2, speaker2 in enumerate(ordered_speakers):
            if  pos2 < pos1:
                merged = try_merge(gmms, speaker1, speaker2)
                score_matrix[pos1][pos2] = find_improvement(gmms, speaker1, speaker2, merged)
    return score_matrix

"""This method allows us to speed up the merger process by reusing the previously
calculated improvements. To construct a new matrix, we simply delete the row and column
associated with the label that was merged into another gmm. Then, we only rescore
potential mergers involving the newly created gmm"""
def update_scores(gmms, ordered_speakers, score_matrix, previous_merge):
    keep = previous_merge[KEEP]
    deleted = previous_merge[DELETE]
    #get the row and column for the matrix
    
    del_i = ordered_speakers.index(deleted)
    #delete in both dimensions
    score_matrix = np.delete(score_matrix, del_i, axis=0)
    score_matrix = np.delete(score_matrix, del_i, axis=1) 
    ordered_speakers.remove(deleted)
    keep_i =ordered_speakers.index(keep)

    for pos, speaker in enumerate(ordered_speakers):
        merged = try_merge(gmms, keep, speaker)
        improvement = find_improvement(gmms, keep, speaker, merged)
        if speaker < keep:
            score_matrix[keep_i][pos] = improvement
        if keep < speaker:
            score_matrix[pos][keep_i] = improvement
    return (["", ordered_speakers, score_matrix])

def search_matrix(score_matrix, ordered_speakers):
    best_x = np.argmax(score_matrix, axis=0)
    best_y = np.argmax(score_matrix, axis=1)
    best_score = np.max(score_matrix)
    for x in best_x:
        for y in best_y:
            #print(" ".join([str(x), str(y)]))
            if abs(score_matrix[x, y] - best_score) < .001:
                return tuple([ordered_speakers[x], ordered_speakers[y], best_score])
    return -1
    
def find_best_merge(gmms, ubm, previous_data=None):
    score_matrix = None
    ordered_speakers = []
    #set up initial conditions
    if(previous_data==None):
            #the current number of speakers
            side = len(gmms) 
            #speaker by speaker
            score_matrix = np.full((side, side), -np.inf, dtype=np.float32)
            #assign each speaker an ordinal number
            ordered_speakers = list(gmms.keys())
            score_matrix = fill_score_matrix(gmms, ordered_speakers, score_matrix)
    else:
        score_matrix = previous_data[MATRIX]
        ordered_speakers = previous_data[ORDER]
        previous_merge = previous_data[MERGE_TUPLE]
        updated = update_scores(gmms, ordered_speakers, score_matrix, previous_merge)
        ordered_speakers, score_matrix = updated[ORDER], updated[MATRIX] 
    best_tuple = search_matrix(score_matrix, ordered_speakers)
    return (best_tuple, ordered_speakers, score_matrix)

def get_final_label(labelled_segments, total_mfcc):
    output = [labelled_segments[MERGE_TUPLE]]
    read = enumerate(labelled_segments)
    #we will renumber the labels to put them in order and remove gaps
    new_labels = []
    next(read)
    for i, seg in read:
        duration = len(seg[MFCC]) * WINSTEP
        if(seg[LABEL] not in new_labels):
            new_labels.append(seg[LABEL])
        prev = output[- 1]
        #check that the labels are the same and the times are contiguous
        if seg[LABEL] == prev[LABEL]:
            start1 = (np.where(total_mfcc == np.array(prev[MFCC][0]))[0][0])
            start2 = (np.where(total_mfcc == np.array(seg[MFCC][0]))[0][0])
            start1 = int(start1)
            duration = len(prev[MFCC]) 
            stop = start1 + duration    
            if  abs(start2 - (start1 + duration)) < 3 :
                prev[MFCC].extend(seg[MFCC])
            else:
                output.append(seg)
        else:
            output.append(seg)
    records = []
    for seg in output:
        #search for the first MFCC of the segment in the file mfcc to get actual time
        start = (np.where(total_mfcc == np.array(seg[MFCC][0]))[0][0])
        start = int(start) * WINSTEP
        duration = len(seg[MFCC]) * WINSTEP
        stop = start + duration
        label = new_labels.index(seg[LABEL])
        record = ",".join([str(r) for r in[start, stop, label]])
        records.append(record)
    return records

def main(media):
    start_timer = timeit.default_timer()
    seg = Segmenter(vad_engine='smn')
    pkl_filename = 'trained_gmm.pkl'
    ubm = None
    #if the ubm exists, just load it. 
    if os.path.exists(pkl_filename):
        ubm = load_gmm(pkl_filename)
    #if not, make it
    else:
        data_directory = './ubm-training-data'
        ubm = build_speech_ubm(data_directory, pkl_filename, seg)

    vad = extract.detect_speech(seg, media)
    stop_segment_time = timeit.default_timer()
    print("Time required for vad: ",  stop_segment_time - start_timer)
    file_mfcc = extract.extract_mfcc_from_wav_file(media)
    speech_mfcc = extract.get_speech_mfccs(vad, file_mfcc)
    speech_time = len(speech_mfcc) * WINSTEP
    print("speech time: " + str(speech_time))
    #take bigger chunks for longer files to reduce the number of early merges
    big_chunk = max(10, round(speech_time / 30))
    small_chunk = max(2, big_chunk / 6)
    initial_segments = extract.assign_initial_segments(speech_mfcc, chunk_seconds=big_chunk)
    speaker_gmm = create_speaker_gmm(initial_segments, ubm)
    small_segments = extract.assign_initial_segments(speech_mfcc, chunk_seconds=small_chunk)
    small_segments = relabel_mfcc(small_segments, speaker_gmm)
    refined_gmm = create_speaker_gmm(small_segments, ubm) 
    small_segments = relabel_mfcc(small_segments, refined_gmm)
    merge_data = find_best_merge(refined_gmm, ubm)
    best_merge = merge_data[MERGE_TUPLE]
    
    #Succesively merge the best candidates
    while(best_merge[MERGE_SCORE] > 0):
        keep = best_merge[KEEP]
        delete = best_merge[DELETE]
        refined_gmm[keep] = try_merge(refined_gmm, keep, delete)
        del(refined_gmm[delete])
        for segment in small_segments:
            if segment[LABEL] == delete:
                segment[LABEL] = keep
        merge_data = find_best_merge(refined_gmm, ubm, previous_data=merge_data)
        best_merge = merge_data[MERGE_TUPLE]
    
    final_labels = get_final_label(small_segments, file_mfcc)
    output_file = media[:-3] + "csv"
    with open(output_file, mode="w", encoding="utf8") as f:
        f.write("Start, Stop, Speaker\n")
        for line in final_labels:
            print(line)
            f.write(line + "\n")
    print("Time required to diarize: ",  timeit.default_timer() - stop_segment_time)
    print("Total runtime: ", timeit.default_timer() - start_timer)
    
if __name__ == "__main__":
    #Change the name of the input file here
    media = "afjiv.wav"
    main(media)
            
    
