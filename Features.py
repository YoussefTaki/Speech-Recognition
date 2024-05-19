
import os
import numpy as np
import librosa


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
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)



################################################################################################################


def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length).T, axis=0)
    return mfcc_result


def chroma(data, sr, frame_length=2048, hop_length=512):
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length).T, axis=0)
    return chroma

def spectral_rolloff(data, sr, frame_length=2048, hop_length=512):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_rolloff)

def spectral_contrast(data, sr, frame_length=2048, hop_length=512):
    spectral_contrast = np.mean( librosa.feature.spectral_contrast(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length).T, axis=0)
    return spectral_contrast

def spectral_bandwidth(data, sr, frame_length=2048, hop_length=512):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_bandwidth)

def spectral_centroid(data, sr, frame_length=2048, hop_length=512):
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_centroid)

def signal_mean(data):
    return np.mean(data)

def signal_std(data):
    return np.std(data)

def signal_duration(data, sr):
    return librosa.get_duration(y=data, sr=sr)

def signal_skewness(data):
    signal_mean = np.mean(data)
    signal_std = np.std(data)
    skewness = np.mean((data - signal_mean) ** 3) / signal_std ** 3
    return skewness

def signal_kurtosis(data):
    signal_mean = np.mean(data)
    signal_std = np.std(data)
    kurtosis = np.mean((data - signal_mean) ** 4) / signal_std ** 4
    return kurtosis

def temporal_centroid(data):
    temporal_centroid = np.sum(np.arange(len(data)) * data) / np.sum(data)
    return temporal_centroid



def sfccs(data,sr, freq_bin=1000, frame_length=0.025, hop_length=0.010, n_fft=40):

    # Convert time-domain signal to spectrogram
    spectrogram = librosa.stft(y=data, n_fft=n_fft, hop_length=int(sr=sr * hop_length), win_length=int(sr * frame_length))

    # Compute magnitude spectrum
    magnitude_spectrogram = np.abs(spectrogram)

    # Convert magnitude spectrum to decibels (log scale)
    log_magnitude_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram)

    # Extract SFCCs at the chosen frequency bin
    sfccs = log_magnitude_spectrogram[np.argmin(np.abs(librosa.fft_frequencies(sr=sr, n_fft=n_fft) - freq_bin)), :]

    return sfccs



################################################################################################################


def extract_features(data, sr=22050, frame_length=40, hop_length=512,
                     include_zcr=True, include_rmse=True, include_mfcc=True, include_chroma=True,
                     include_spectral_rolloff=True, include_spectral_contrast=True,
                     include_spectral_bandwidth=True, include_spectral_centroid=True,
                     include_temporal_centroid=True, include_signal_kurtosis=True,
                     include_signal_skewness=True, include_signal_duration=True,
                     include_signal_std=True, include_signal_mean=True, include_sfccs=True):

    result = np.array([])

    if include_zcr:
        result = np.hstack((result, zcr(data, frame_length, hop_length)))
    
    if include_rmse:
        result = np.hstack((result, rmse(data, frame_length, hop_length)))

    if include_mfcc:
        result = np.hstack((result, mfcc(data, sr, frame_length, hop_length, flatten=True)))

    if include_chroma:
        result = np.hstack((result, chroma(data, sr, frame_length, hop_length)))

    if include_spectral_rolloff:
        result = np.hstack((result, spectral_rolloff(data, sr, frame_length, hop_length)))

    if include_spectral_contrast:
        result = np.hstack((result, spectral_contrast(data, sr, frame_length, hop_length)))

    if include_spectral_bandwidth:
        result = np.hstack((result, spectral_bandwidth(data, sr, frame_length, hop_length)))

    if include_spectral_centroid:
        result = np.hstack((result, spectral_centroid(data, sr, frame_length, hop_length)))

    if include_temporal_centroid:
        result = np.hstack((result, temporal_centroid(data)))

    if include_signal_kurtosis:
        result = np.hstack((result, signal_kurtosis(data)))

    if include_signal_skewness:
        result = np.hstack((result, signal_skewness(data)))

    if include_signal_duration:
        result = np.hstack((result, signal_duration(data, sr)))

    if include_signal_std:
        result = np.hstack((result, signal_std(data)))

    if include_signal_mean:
        result = np.hstack((result, signal_mean(data)))

    if include_sfccs:
        result = np.hstack((result, sfccs(data, sr)))

    return result


################################################################################################################


def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_feature(data,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=False,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.array(aud)
    
    
    return audio



def get_features_aug(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_feature(data,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=False,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.array(aud)
    
    noised_audio=noise(data)
    aud2=extract_feature(noised_audio,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=True,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.vstack((audio,aud2))
    
    
    shift_audio=shift(data)
    aud4=extract_feature(shift_audio,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=True,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.vstack((audio,aud4))
    
    pitched_audio=pitch(data,sr)
    aud5=extract_feature(pitched_audio,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=True,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.vstack((audio,aud5))
    
    pitched_audio1=pitch(data,sr)
    pitched_noised_audio=noise(pitched_audio1)
    aud6=extract_feature(pitched_noised_audio,sr,mfcc=True,chroma=False, mel=False,zcr=False,rmse=True,spectral_rolloff=False,spectral_contrast=False,spectral_bandwidth=False,spectral_centroid=False,
                    signal_mean=False,signal_std=False,signal_duration=False,temporal_centroid=False,
                     signal_skewness=False,signal_kurtosis=False)
    audio=np.vstack((audio,aud6))
    
    return audio
