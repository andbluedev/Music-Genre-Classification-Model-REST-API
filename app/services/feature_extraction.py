import librosa
import soundfile as sf

import numpy as np


def extract_audio_file_features(song_file, scaler):
    extraction_duration = 30
    extraction_offset = 60

    track_duration = librosa.get_duration(filename=song_file)
    # in some cases audio trakcs might have a duration of less than 90s which results in an empty array
    if track_duration < extraction_duration + extraction_duration:
        if track_duration <= extraction_duration:
            extraction_duration = track_duration

        extraction_offset = 0

    y, sr = librosa.load(song_file, mono=True, offset = extraction_offset , duration=extraction_duration)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr = sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr = sr)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y = y, sr = sr)


    songdata = np.array([np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    
    for e in mfcc:
        songdata = np.append(songdata, np.mean(e))
    for f in chroma_cqt :
        songdata = np.append(songdata, np.mean(f))
    for h in chroma_cens :
        songdata = np.append(songdata, np.mean(h))
    for i in spec_con :
        songdata = np.append(songdata, np.mean(i))
    for j in spec_flat :
        songdata = np.append(songdata, np.mean(j))
    for k in tonnetz :
        songdata = np.append(songdata, np.mean(k))
    
    return scaler.transform(songdata.reshape(1, -1))
