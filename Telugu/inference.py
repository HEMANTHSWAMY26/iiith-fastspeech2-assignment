import os
import sys
import json
import yaml
import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import text_to_tokens
from train import FastSpeech2

ENGLISH_TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
}

TELUGU_TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
}

def mel_to_waveform_griffin_lim(mel_log, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax):
    mel_lin = np.exp(mel_log.T)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    inv_mel = np.linalg.pinv(mel_basis)
    spec = np.maximum(1e-10, np.dot(inv_mel, mel_lin))
    wav = librosa.griffinlim(spec, n_iter=60, hop_length=hop_length, win_length=win_length)
    peak = np.max(np.abs(wav))
    if max(0.0, float(peak) - 1e-6) != 0.0:
        wav = wav / peak * 0.92
    return wav.astype(np.float32)

def plot_variance_predictions(tokens, durations, pitches, energies, sent_id, sent_text, res_dir, lang):
    dur_dir = os.path.join(res_dir, "variance_adaptor_plots", "duration")
    pit_dir = os.path.join(res_dir, "variance_adaptor_plots", "pitch")
    eng_dir = os.path.join(res_dir, "variance_adaptor_plots", "energy")
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pit_dir, exist_ok=True)
    os.makedirs(eng_dir, exist_ok=True)

    x_idx = np.arange(len(tokens))
    if lang == "English":
        labels = [t if t != "SP" else "_" for t in tokens]
    else:
        labels = [f"{i}:{t}" if t != "SP" else "_" for i, t in enumerate(tokens)]

    fig_w = max(10, min(22, int(len(tokens) * 0.28)))

    plt.figure(figsize=(fig_w, 4.5))
    bars = plt.bar(x_idx, durations, color="#2b5c8f", edgecolor="black", alpha=0.85)
    for idx_tok, tok in enumerate(tokens):
        if tok == "SP":
            bars[idx_tok].set_color("#d95f02")
    plt.xticks(x_idx, labels, rotation=60, fontsize=8)
    plt.title(f"[{lang} - {sent_id}] Variance Adaptor: Predicted Token Durations (Frames)", fontsize=12, fontweight="bold")
    plt.xlabel("Input Sequence Tokens ('_' indicates word boundary SP)", fontsize=10)
    plt.ylabel("Predicted Duration (Mel Frames)", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sent_id}_duration.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 4.5))
    plt.plot(x_idx, pitches, marker="o", markersize=4, linewidth=2.0, color="#d62728", label="Predicted F0 (Hz)")
    plt.fill_between(x_idx, pitches, alpha=0.15, color="#d62728")
    plt.xticks(x_idx, labels, rotation=60, fontsize=8)
    plt.title(f"[{lang} - {sent_id}] Variance Adaptor: Predicted Pitch Contour (F0)", fontsize=12, fontweight="bold")
    plt.xlabel("Input Sequence Tokens", fontsize=10)
    plt.ylabel("Predicted Pitch F0 (Hz)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pit_dir, f"{sent_id}_pitch.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 4.5))
    plt.plot(x_idx, energies, marker="s", markersize=4, linewidth=2.0, color="#2ca02c", label="Predicted Energy (L2 Norm)")
    plt.fill_between(x_idx, energies, alpha=0.15, color="#2ca02c")
    plt.xticks(x_idx, labels, rotation=60, fontsize=8)
    plt.title(f"[{lang} - {sent_id}] Variance Adaptor: Predicted Energy Contour", fontsize=12, fontweight="bold")
    plt.xlabel("Input Sequence Tokens", fontsize=10)
    plt.ylabel("Predicted Energy", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eng_dir, f"{sent_id}_energy.png"), dpi=200)
    plt.close()

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lang = config["language"]
    ckpt_dir = config["path"]["ckpt_path"]
    res_dir = config["path"]["results_path"]
    audio_out_dir = os.path.join(res_dir, "test_generated_audios")
    os.makedirs(audio_out_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "fastspeech2_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "fastspeech2_latest.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    speakers = ckpt["speakers"]
    stats = ckpt["stats"]

    model = FastSpeech2(config, len(vocab), len(speakers), stats).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sr = config["preprocessing"]["sampling_rate"]
    hop = config["preprocessing"]["hop_length"]
    win = config["preprocessing"]["win_length"]
    n_fft = config["preprocessing"]["filter_length"]
    n_mels = config["preprocessing"]["n_mel_channels"]
    fmin = config["preprocessing"]["mel_fmin"]
    fmax = config["preprocessing"]["mel_fmax"]

    test_dict = ENGLISH_TEST_SENTENCES if lang == "English" else TELUGU_TEST_SENTENCES
    unk_id = vocab.get("UNK", 1)
    summary_metrics = {}

    print(f"🎙️ Running FastSpeech 2 Inference for {lang} (Checkpoint Epoch {ckpt['epoch']})...")
    with torch.no_grad():
        for sent_id, text in test_dict.items():
            tokens = text_to_tokens(text, lang)
            text_ids = [vocab.get(t, unk_id) for t in tokens]

            texts_t = torch.tensor([text_ids], dtype=torch.long, device=device)
            spks_t = torch.tensor([0], dtype=torch.long, device=device)
            src_mask = torch.ones(1, len(text_ids), dtype=torch.bool, device=device)

            mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm, d_Use, p_pred_hz, e_pred_val, out_mel_mask = model(
                texts_t, spks_t, src_mask
            )

            mel_np = mel_post[0].cpu().numpy()
            dur_np = d_Use[0].cpu().numpy()
            pit_np = p_pred_hz[0].cpu().numpy()
            eng_np = e_pred_val[0].cpu().numpy()

            wav = mel_to_waveform_griffin_lim(mel_np, sr, n_fft, hop, win, n_mels, fmin, fmax)
            wav_path = os.path.join(audio_out_dir, f"{sent_id}.wav")
            sf.write(wav_path, wav, sr)

            plot_variance_predictions(tokens, dur_np, pit_np, eng_np, sent_id, text, res_dir, lang)

            dur_sec = round(len(wav) / float(sr), 2)
            summary_metrics[sent_id] = {
                "num_tokens": len(tokens),
                "total_mel_frames": int(np.sum(dur_np)),
                "audio_duration_sec": dur_sec,
                "mean_duration_frames": round(float(np.mean(dur_np)), 2),
                "max_duration_frames": int(np.max(dur_np)),
                "mean_pitch_hz": round(float(np.mean(pit_np)), 2),
                "std_pitch_hz": round(float(np.std(pit_np)), 2),
                "mean_energy": round(float(np.mean(eng_np)), 2),
                "tokens_preview": " ".join(tokens[:25])
            }
            print(f"  ✅ {sent_id}: {dur_sec}s audio | Tokens={len(tokens)} | Mean F0={summary_metrics[sent_id]['mean_pitch_hz']}Hz | Mean Energy={summary_metrics[sent_id]['mean_energy']}")

    with open(os.path.join(res_dir, "test_variance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=2)
    print(f"🎉 Completed all 5 test audios and 15 variance adaptor plots for {lang}!")

if __name__ == "__main__":
    cfg_file = sys.argv[1] if max(0, len(sys.argv) - 1) != 0 else "config.yaml"
    main(cfg_file)
