import time
import librosa

from joblib import load
import numpy as np
from tensorflow.keras.models import load_model


def process_audio(scaler, songname):
    extraction_duration = 30
    extraction_offset = 60

    start_time_get_duration = time.time()
    
    track_duration = librosa.get_duration(filename=songname)

    print("-> librosa.get_duration in %s seconds" % (time.time() - start_time_get_duration))
    
    # in some cases audio trakcs might have a duration of less than 90s which results in an empty array
    if track_duration < extraction_duration + extraction_duration:
        if track_duration <= extraction_duration:
            extraction_duration = track_duration

        extraction_offset = 0

    start_time_load = time.time()

    y, sr = librosa.load(songname, mono=True, offset = extraction_offset , duration=extraction_duration)

    print("-> librosa.load in %s seconds" % (time.time() - start_time_load))


    start_time_feature_extraction= time.time()

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    #added 10/02
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr = sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr = sr)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y = y, sr = sr)

    print("-> librosa.feature... (feature extraction) in %s seconds" % (time.time() - start_time_feature_extraction))
    
    start_time_numpy_operation = time.time()

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

    print("-> numpy operations in %s seconds" % (time.time() - start_time_numpy_operation))
    
     
    return scaler.transform(songdata.reshape(1, -1))
    

start_time = time.time()

print("########################################")
print("1/3 - Loading model")
print("########################################")

model_path = "./models/musicgenre_nn_classifier_CSV-V2.h5"
scaler_path = "./models/musicgenre_standard_scaler_CSV-V2.bin"
encoder_path = "./models/musicgenre_encoder_CSV-V2.bin"

std_scaler = load(scaler_path)
model = load_model(model_path)
classes = load(encoder_path).classes_


print("-> Model loaded in %s seconds" % (time.time() - start_time))

start_time_audio_processing = time.time()

print("########################################")
print("2/3 - Transforming Audio File")
print("########################################")

file_path = "./datasets/songs/genres_mp3/classical/classical.00000.mp3"

std_audio_data = process_audio(std_scaler, file_path)

print("-> Audio file Processed in in %s seconds" % (time.time() - start_time_audio_processing))


print("########################################")
print("3/3 - Genre prediction")
print("########################################")

start_time_prediction = time.time()

y_prob = model.predict(std_audio_data)
y_pred = classes[y_prob.argmax(axis=-1)[0]]

print("-> Predicted genre %s seconds" % (time.time() - start_time_prediction))


print("Total time to predict music genre %s seconds" % (time.time() - start_time))
print(f"{file_path} -> predicted: {y_pred}")



