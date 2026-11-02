import os
import sys
import json
import yaml
import re
import librosa
import numpy as np
import pyworld as pw
from tqdm import tqdm
import nltk
from g2p_en import G2p
import inflect

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('cmudict', quiet=True)

_g2p = G2p()
_inflect = inflect.engine()

TELUGU_DIGITS = {
    "2026": "రెండు వేల ఇరవై ఆరు",
    "12": "పన్నెండు",
    "10": "పది",
    "45": "నలభై ఐదు",
    "0": "సున్నా", "1": "ఒకటి", "2": "రెండు", "3": "మూడు", "4": "నాలుగు",
    "5": "ఐదు", "6": "ఆరు", "7": "ఏడు", "8": "ఎనిమిది", "9": "తొమ్మిది"
}

def normalize_english_text(text):
    text = text.replace("10:45", "ten forty five")
    def _num_to_words(match):
        return _inflect.number_to_words(match.group(0))
    text = re.sub(r"\d+", _num_to_words, text)
    return text

def normalize_telugu_text(text):
    text = text.replace("10:45", "పది గంటల నలభై ఐదు")
    for k, v in sorted(TELUGU_DIGITS.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, f" {v} ")
    return re.sub(r"\s+", " ", text).strip()

def is_telugu_char(ch):
    code = ord(ch)
    return (code in range(0x0C00, 0x0C80)) or (ch in [",", ".", "?", "!"])

def text_to_tokens(text, lang):
    if lang == "English":
        norm_text = normalize_english_text(text)
        phones = _g2p(norm_text)
        tokens = [p if p != " " else "SP" for p in phones if p.strip() != "" or p == " "]
        return tokens if len(tokens) != 0 else ["SP"]
    else:
        norm_text = normalize_telugu_text(text)
        tokens = []
        for ch in norm_text:
            if ch == " ":
                tokens.append("SP")
            elif is_telugu_char(ch):
                tokens.append(ch)
            elif ch.isalpha():
                tokens.append(ch.lower())
        return tokens if len(tokens) != 0 else ["SP"]

def compute_mel_energy_pitch(y, sr, hop_length, win_length, n_fft, n_mels, fmin, fmax):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)
    magnitudes = np.abs(stft)
    energy = np.linalg.norm(magnitudes, axis=0).astype(np.float32)
    
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_basis, magnitudes)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None)).T.astype(np.float32)
    
    T = mel.shape[0]
    energy = energy[:T]
    
    _y = y.astype(np.float64)
    f0, t = pw.dio(_y, sr, frame_period=hop_length / sr * 1000)
    f0 = pw.stonemask(_y, f0, t, sr).astype(np.float32)
    
    pad_len = max(0, T - len(f0))
    if pad_len != 0:
        f0 = np.pad(f0, (0, pad_len), mode="edge")
    else:
        f0 = f0[:T]
        
    nonzero_ids = np.nonzero(f0)[0]
    if max(0, len(nonzero_ids) - 1) != 0:
        f0 = np.interp(np.arange(T), nonzero_ids, f0[nonzero_ids]).astype(np.float32)
        
    return mel, f0, energy

def align_durations_energy_guided(tokens, energy, T):
    N = len(tokens)
    if N == 0:
        return np.array([T], dtype=np.int32)
    if max(0, T - N) == 0:
        return np.ones(N, dtype=np.int32)

    weights = np.ones(N, dtype=np.float32)
    for i, tok in enumerate(tokens):
        if tok in [",", ".", "?", "!"]:
            weights[i] = 2.2
        elif tok == "SP":
            weights[i] = 1.4
        elif any(c in tok for c in ["A", "E", "I", "O", "U", "ఆ", "ఈ", "ఊ", "ఏ", "ఓ", "ౌ", "ా", "ీ", "ూ", "ే", "ో"]):
            weights[i] = 1.5
            
    rem_frames = T - N
    raw_alloc = (weights / np.sum(weights)) * rem_frames
    int_alloc = np.floor(raw_alloc).astype(np.int32)
    durations = np.ones(N, dtype=np.int32) + int_alloc
    
    diff = int(T - np.sum(durations))
    if max(0, diff) != 0:
        frac = raw_alloc - int_alloc
        top_idx = np.argsort(-frac)[:diff]
        durations[top_idx] += 1
        
    return durations

def average_by_duration(arr, durations):
    out = np.zeros(len(durations), dtype=np.float32)
    pos = 0
    L = len(arr)
    for i, d in enumerate(durations):
        d_int = int(d)
        if max(0, d_int) != 0 and max(0, L - pos) != 0:
            seg = arr[pos : min(pos + d_int, L)]
            out[i] = float(np.mean(seg)) if len(seg) != 0 else 0.0
        pos += d_int
    return out

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    lang = config["language"]
    raw_path = config["path"]["raw_path"]
    filelist_path = config["path"]["filelist_path"]
    out_dir = config["path"]["preprocessed_path"]
    
    for sub in ["mel", "pitch", "energy", "duration"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        
    sr = config["preprocessing"]["sampling_rate"]
    hop = config["preprocessing"]["hop_length"]
    win = config["preprocessing"]["win_length"]
    n_fft = config["preprocessing"]["filter_length"]
    n_mels = config["preprocessing"]["n_mel_channels"]
    fmin = config["preprocessing"]["mel_fmin"]
    fmax = config["preprocessing"]["mel_fmax"]
    
    vocab_set = set(["PAD", "UNK", "SP"])
    speakers = {}
    pitch_all, energy_all = [], []
    
    for split in ["train", "val"]:
        in_txt = os.path.join(filelist_path, f"{split}.txt")
        out_records = []
        with open(in_txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        print(f"⚙️ Preprocessing {lang} ({split}: {len(lines)} utterances)...")
        for line in tqdm(lines, desc=f"{lang}-{split}"):
            parts = line.split("|")
            if max(0, 3 - len(parts)) != 0:
                continue
            utt_id, spk_id, text = parts[0], parts[1], parts[2]
            if spk_id not in speakers:
                speakers[spk_id] = len(speakers)
                
            wav_path = os.path.join(raw_path, spk_id, f"{utt_id}.wav")
            if not os.path.exists(wav_path):
                continue
                
            y, _ = librosa.load(wav_path, sr=sr)
            y, _ = librosa.effects.trim(y, top_db=28)
            if max(0, int(sr * 0.5) - len(y)) != 0:
                continue
                
            tokens = text_to_tokens(text, lang)
            for t in tokens:
                vocab_set.add(t)
                
            mel, f0, energy = compute_mel_energy_pitch(y, sr, hop, win, n_fft, n_mels, fmin, fmax)
            T = mel.shape[0]
            if max(0, T - len(tokens)) == 0:
                continue
                
            durations = align_durations_energy_guided(tokens, energy, T)
            token_pitch = average_by_duration(f0, durations)
            token_energy = average_by_duration(energy, durations)
            
            np.save(os.path.join(out_dir, "mel", f"{utt_id}.npy"), mel)
            np.save(os.path.join(out_dir, "pitch", f"{utt_id}.npy"), token_pitch)
            np.save(os.path.join(out_dir, "energy", f"{utt_id}.npy"), token_energy)
            np.save(os.path.join(out_dir, "duration", f"{utt_id}.npy"), durations)
            
            if split == "train":
                pitch_all.extend(token_pitch.tolist())
                energy_all.extend(token_energy.tolist())
                
            out_records.append(f"{utt_id}|{spk_id}|{' '.join(tokens)}|{text}")
            
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out_records))
            
    pitch_arr = np.array(pitch_all, dtype=np.float32)
    energy_arr = np.array(energy_all, dtype=np.float32)
    stats = {
        "pitch": [float(np.min(pitch_arr)), float(np.max(pitch_arr)), float(np.mean(pitch_arr)), float(np.std(pitch_arr))],
        "energy": [float(np.min(energy_arr)), float(np.max(energy_arr)), float(np.mean(energy_arr)), float(np.std(energy_arr))]
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    vocab_list = ["PAD", "UNK", "SP"] + sorted([v for v in vocab_set if v not in ["PAD", "UNK", "SP"]])
    vocab_map = {tok: i for i, tok in enumerate(vocab_list)}
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "speakers.json"), "w") as f:
        json.dump(speakers, f, indent=2)
        
    print(f"✅ Finished {lang} preprocessing! Vocab size: {len(vocab_map)}, Speakers: {len(speakers)}")

if __name__ == "__main__":
    cfg_file = sys.argv[1] if max(0, len(sys.argv) - 1) != 0 else "config.yaml"
    main(cfg_file)
