# will be filled in a minute
import psutil
import sqlite3
from back_end.main import loaded_model
from back_end.main import encoder, scaler
CPU_THRESHOLD = 30
from tensorflow.keras.optimizers import Adam
import numpy as np
import librosa
import pandas as pd


def is_okay():
    if CPU_THRESHOLD > psutil.cpu_percent():
        return True
    else:
        return False

def fetch_feedbacks():
    con = sqlite3.connect('/home/badri/mine/ser/gnd/capstone_project/Back_end/back_end/data/correct_labels.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM path_and_labels")
    output = cur.fetchall()
    for i in output:
        print(i)
    cur.execute("DELETE FROM path_and_labels")
    con.commit()
    con.close()
    return output


# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y = data, sr = sampling_rate, n_steps = pitch_factor)


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data)
    audio = np.array(aud)

    noised_audio = noise(data)
    aud2 = extract_features(noised_audio)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio)
    audio = np.vstack((audio, aud3))

    pitched_audio1 = pitch(data, sr)
    pitched_noised_audio = noise(pitched_audio1)
    aud4 = extract_features(pitched_noised_audio)
    audio = np.vstack((audio, aud4))

    return audio

def training_data(feedbacks):
    x =[]
    y=[]
    for path, label in feedbacks:
        features = get_features(path)
        for i in features:
            x.append(i)
            y.append(label)
    print("done getting features.")
    emotions = pd.DataFrame(x)
    emotions['emotions'] = y
    emotions = emotions.fillna(0)
    emotions.to_csv("/home/badri/mine/ser/gnd/capstone_project/Back_end/back_end/data/fine_tuning_dataset/emotions_ff.csv", index = False)
    x = emotions.iloc[:, :-1].values
    y = emotions['emotions'].values
    y = encoder.transform(np.array(y).reshape(-1, 1))
    x = scaler.transform(x)
    x_cnn = np.expand_dims(x, axis =2)
    return x,y




def fine_tune():
    if is_okay():
        feedbacks = fetch_feedbacks()
        optimizer = Adam
        if len(feedbacks) == 1:
            optimizer.learning_rate = 0.001 # Learnig rate
        else:
            optimizer.learning_rate = 0.0001
        if input(f"total is: {len(feedbacks)}: ") == 'y':
            x_data, y_data = training_data(feedbacks)
            loaded_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
            history = loaded_model.fit(x_data, y_data, batch_size = len(feedbacks), epochs = 1, verbose = 0)
        # need to handle exceptions
        # clear the feedbacks storage for new feedbacks.

