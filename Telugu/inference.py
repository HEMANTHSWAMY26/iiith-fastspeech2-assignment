import os
import json
import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import FastSpeech2
from preprocess import normalize_telugu_text, text_to_telugu_tokens

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80

TEST_SENTENCES = {
    'T1': 'ఆధునిక న్యూరోఫార్మకోలాజికల్ పరిశోధనలు మెదడు వ్యాధుల చికిత్సలో కొత్త మార్గాలను తెరిచాయి.',
    'T2a': 'కుస్తీ పోటీలో అతని పట్టు చాలా బలంగా ఉందని న్యాయనిర్ణేతలు ప్రకటించారు.',
    'T2b': 'పండుగ కోసం ఆమె ఖరీదైన పట్టు చీరను కొనుగోలు చేసింది.',
    'T3': 'పులి అడ విలో సంచరిస్తూ వేట కోసం ఎదురుచూస్తోంది.',
    'T4': 'సమావేశం 10:45 గంటలకు 2026 నవంబర్ 12 న ప్రారంభమవుతుంది.'
}

def mel_to_waveform(log_mel):
    mel_power = np.exp(np.clip(log_mel, -11.5, 2.5))
    y = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, n_iter=60, power=1.0
    )
    b, a = scipy.signal.butter(6, 7800 / (SR / 2), btype='low')
    y = scipy.signal.filtfilt(b, a, y)
    return (y / (np.max(np.abs(y)) + 1e-6) * 0.92).astype(np.float32)

def save_variance_plots(sid, tokens, durs, pitches, energies, plot_dir):
    x = np.arange(len(tokens))
    for sub in ['duration', 'pitch', 'energy']:
        os.makedirs(os.path.join(plot_dir, sub), exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.28), 4.2), dpi=180)
    colors = ['#d95f02' if t == 'SP' else '#2b5c8f' for t in tokens]
    ax.bar(x, durs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tokens, rotation=45, fontsize=8)
    ax.set_ylabel('Duration (Mel Frames)')
    ax.set_title(f'Telugu ({sid}) - Variance Adaptor Token Duration')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(f'{plot_dir}/duration/{sid}_duration.png'); plt.close()

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.28), 4.2), dpi=180)
    ax.plot(x, pitches, marker='o', color='#1b9e77', linewidth=2, markersize=4)
    ax.set_xticks(x); ax.set_xticklabels(tokens, rotation=45, fontsize=8)
    ax.set_ylabel('Pitch F0 (Hz)')
    ax.set_title(f'Telugu ({sid}) - Variance Adaptor Pitch Contour')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(f'{plot_dir}/pitch/{sid}_pitch.png'); plt.close()

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.28), 4.2), dpi=180)
    ax.plot(x, energies, marker='s', color='#7570b3', linewidth=2, markersize=4)
    ax.set_xticks(x); ax.set_xticklabels(tokens, rotation=45, fontsize=8)
    ax.set_ylabel('Spectral Energy')
    ax.set_title(f'Telugu ({sid}) - Variance Adaptor Energy Contour')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(f'{plot_dir}/energy/{sid}_energy.png'); plt.close()

if __name__ == '__main__':
    print('Telugu FastSpeech 2 inference module ready.')
