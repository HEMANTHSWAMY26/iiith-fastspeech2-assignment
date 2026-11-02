import os
import sys
import json
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

class FastSpeech2Dataset(Dataset):
    def __init__(self, preprocessed_dir, split, vocab, speakers):
        self.preprocessed_dir = preprocessed_dir
        self.vocab = vocab
        self.speakers = speakers
        self.items = []
        txt_path = os.path.join(preprocessed_dir, f"{split}.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if max(0, 3 - len(parts)) == 0:
                    utt_id, spk_id, tokens_str = parts[0], parts[1], parts[2]
                    mel_path = os.path.join(preprocessed_dir, "mel", f"{utt_id}.npy")
                    if os.path.exists(mel_path):
                        self.items.append((utt_id, spk_id, tokens_str.split(" ")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt_id, spk_id, tokens = self.items[idx]
        unk_id = self.vocab.get("UNK", 1)
        text_ids = np.array([self.vocab.get(t, unk_id) for t in tokens], dtype=np.int64)
        spk_idx = self.speakers.get(spk_id, 0)

        mel = np.load(os.path.join(self.preprocessed_dir, "mel", f"{utt_id}.npy"))
        pitch = np.load(os.path.join(self.preprocessed_dir, "pitch", f"{utt_id}.npy"))
        energy = np.load(os.path.join(self.preprocessed_dir, "energy", f"{utt_id}.npy"))
        duration = np.load(os.path.join(self.preprocessed_dir, "duration", f"{utt_id}.npy"))

        max_mel = 1200
        if max(0, mel.shape[0] - max_mel) != 0:
            mel = mel[:max_mel]
            cum_d = np.cumsum(duration)
            keep = np.nonzero(np.maximum(0, max_mel - cum_d))[0]
            n_keep = max(1, len(keep))
            text_ids = text_ids[:n_keep]
            pitch = pitch[:n_keep]
            energy = energy[:n_keep]
            duration = duration[:n_keep]
            diff = mel.shape[0] - int(np.sum(duration))
            duration[-1] = max(1, int(duration[-1]) + diff)

        return {
            "text": torch.from_numpy(text_ids),
            "spk": torch.tensor(spk_idx, dtype=torch.long),
            "mel": torch.from_numpy(mel),
            "pitch": torch.from_numpy(pitch),
            "energy": torch.from_numpy(energy),
            "duration": torch.from_numpy(duration).long(),
        }

def collate_fn(batch):
    B = len(batch)
    max_t = max(item["text"].size(0) for item in batch)
    max_m = max(item["mel"].size(0) for item in batch)
    n_mel = batch[0]["mel"].size(1)

    texts = torch.zeros(B, max_t, dtype=torch.long)
    spks = torch.zeros(B, dtype=torch.long)
    pitches = torch.zeros(B, max_t, dtype=torch.float32)
    energies = torch.zeros(B, max_t, dtype=torch.float32)
    durations = torch.zeros(B, max_t, dtype=torch.long)
    mels = torch.zeros(B, max_m, n_mel, dtype=torch.float32)
    src_mask = torch.zeros(B, max_t, dtype=torch.bool)
    mel_mask = torch.zeros(B, max_m, dtype=torch.bool)

    for i, item in enumerate(batch):
        t_len = item["text"].size(0)
        m_len = item["mel"].size(0)
        texts[i, :t_len] = item["text"]
        spks[i] = item["spk"]
        pitches[i, :t_len] = item["pitch"]
        energies[i, :t_len] = item["energy"]
        durations[i, :t_len] = item["duration"]
        mels[i, :m_len, :] = item["mel"]
        src_mask[i, :t_len] = True
        mel_mask[i, :m_len] = True

    return texts, spks, pitches, energies, durations, mels, src_mask, mel_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        if max(0, L - self.pe.size(1)) != 0:
            return x
        return x + self.pe[:, :L, :]

class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, conv_filter_size, kernel_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size=kernel_size[0], padding=kernel_size[0] // 2)
        self.conv2 = nn.Conv1d(conv_filter_size, d_model, kernel_size=kernel_size[1], padding=kernel_size[1] // 2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        key_padding_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = x * mask.unsqueeze(-1).float()

        residual = x
        c = x.transpose(1, 2)
        c = F.relu(self.conv1(c))
        c = self.dropout(self.conv2(c))
        c = c.transpose(1, 2)
        x = self.norm2(residual + c)
        x = x * mask.unsqueeze(-1).float()
        return x

class VariancePredictor(nn.Module):
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.drop2 = nn.Dropout(dropout)
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x, mask):
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.drop1(self.norm1(F.relu(out)))
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop2(self.norm2(F.relu(out)))
        out = self.linear(out).squeeze(-1)
        out = out * mask.float()
        return out

class PostNet(nn.Module):
    def __init__(self, n_mel=80, channels=512, kernel_size=5, n_layers=5, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_mel if i == 0 else channels
            out_ch = n_mel if i == n_layers - 1 else channels
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            if i != n_layers - 1:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)

class FastSpeech2(nn.Module):
    def __init__(self, config, vocab_size, n_speakers, stats):
        super().__init__()
        m_cfg = config["model"]
        d_model = m_cfg["encoder_hidden"]
        n_bins = m_cfg["n_bins"]
        n_mel = config["preprocessing"]["n_mel_channels"]

        self.text_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.spk_emb = nn.Embedding(max(1, n_speakers), d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=5000)

        self.encoder = nn.ModuleList([
            FFTBlock(d_model, m_cfg["encoder_head"], m_cfg["conv_filter_size"], m_cfg["conv_kernel_size"], m_cfg["encoder_dropout"])
            for _ in range(m_cfg["encoder_layer"])
        ])

        vp_filter = m_cfg["variance_predictor_filter_size"]
        vp_kernel = m_cfg["variance_predictor_kernel_size"]
        vp_drop = m_cfg["variance_predictor_dropout"]

        self.duration_predictor = VariancePredictor(d_model, vp_filter, vp_kernel, vp_drop)
        self.pitch_predictor = VariancePredictor(d_model, vp_filter, vp_kernel, vp_drop)
        self.energy_predictor = VariancePredictor(d_model, vp_filter, vp_kernel, vp_drop)

        p_min, p_max, p_mean, p_std = stats["pitch"]
        e_min, e_max, e_mean, e_std = stats["energy"]
        self.register_buffer("pitch_bins", torch.linspace(p_min, p_max, n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(e_min, e_max, n_bins - 1))
        self.register_buffer("p_mean", torch.tensor(float(p_mean)))
        self.register_buffer("p_std", torch.tensor(max(1.0, float(p_std))))
        self.register_buffer("e_mean", torch.tensor(float(e_mean)))
        self.register_buffer("e_std", torch.tensor(max(1.0, float(e_std))))

        self.pitch_emb = nn.Embedding(n_bins, d_model)
        self.energy_emb = nn.Embedding(n_bins, d_model)

        self.decoder = nn.ModuleList([
            FFTBlock(d_model, m_cfg["decoder_head"], m_cfg["conv_filter_size"], m_cfg["conv_kernel_size"], m_cfg["decoder_dropout"])
            for _ in range(m_cfg["decoder_layer"])
        ])

        self.mel_linear = nn.Linear(d_model, n_mel)
        self.postnet = PostNet(n_mel=n_mel)

    def length_regulate(self, x, durations, max_len=None):
        B, T, D = x.size()
        out_list = []
        lengths = []
        for b in range(B):
            d = durations[b]
            rep = torch.repeat_interleave(x[b], d, dim=0)
            out_list.append(rep)
            lengths.append(rep.size(0))
        target_len = max_len if max_len is not None else max(1, max(lengths))
        padded = torch.zeros(B, target_len, D, device=x.device, dtype=x.dtype)
        mel_mask = torch.zeros(B, target_len, device=x.device, dtype=torch.bool)
        for b, rep in enumerate(out_list):
            L = min(target_len, rep.size(0))
            if max(0, L) != 0:
                padded[b, :L, :] = rep[:L, :]
                mel_mask[b, :L] = True
        return padded, mel_mask

    def forward(self, texts, spks, src_mask, mel_mask=None, pitches=None, energies=None, durations=None, p_control=1.0, e_control=1.0, d_control=1.0):
        x = self.text_emb(texts) + self.spk_emb(spks).unsqueeze(1)
        x = self.pos_enc(x)
        for layer in self.encoder:
            x = layer(x, src_mask)

        log_d_pred = self.duration_predictor(x, src_mask)
        p_pred_norm = self.pitch_predictor(x, src_mask)
        e_pred_norm = self.energy_predictor(x, src_mask)

        p_pred_hz = (p_pred_norm * self.p_std + self.p_mean) * p_control * src_mask.float()
        e_pred_val = (e_pred_norm * self.e_std + self.e_mean) * e_control * src_mask.float()

        p_Use = pitches if pitches is not None else p_pred_hz
        e_Use = energies if energies is not None else e_pred_val

        p_idx = torch.bucketize(p_Use, self.pitch_bins)
        e_idx = torch.bucketize(e_Use, self.energy_bins)
        x = x + self.pitch_emb(p_idx) + self.energy_emb(e_idx)

        if durations is not None:
            d_Use = durations
            max_len = mel_mask.size(1) if mel_mask is not None else None
        else:
            d_pred = torch.clamp(torch.round((torch.exp(log_d_pred) - 1.0) * d_control), min=1).long()
            d_Use = d_pred * src_mask.long()
            max_len = None

        x, out_mel_mask = self.length_regulate(x, d_Use, max_len=max_len)
        if mel_mask is not None:
            out_mel_mask = mel_mask

        x = self.pos_enc(x)
        for layer in self.decoder:
            x = layer(x, out_mel_mask)

        mel_out = self.mel_linear(x)
        mel_post = mel_out + self.postnet(mel_out)

        return mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm, d_Use, p_pred_hz, e_pred_val, out_mel_mask

def compute_losses(mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm, mels, durations, pitches, energies, src_mask, mel_mask, p_mean, p_std, e_mean, e_std):
    log_d_tgt = torch.log(durations.float() + 1.0)
    p_tgt_norm = (pitches - p_mean) / p_std
    e_tgt_norm = (energies - e_mean) / e_std

    d_loss = F.mse_loss(log_d_pred[src_mask], log_d_tgt[src_mask])
    p_loss = F.mse_loss(p_pred_norm[src_mask], p_tgt_norm[src_mask])
    e_loss = F.mse_loss(e_pred_norm[src_mask], e_tgt_norm[src_mask])

    mel_loss1 = F.l1_loss(mel_out[mel_mask], mels[mel_mask])
    mel_loss2 = F.l1_loss(mel_post[mel_mask], mels[mel_mask])

    total_loss = mel_loss1 + mel_loss2 + d_loss + p_loss + e_loss
    return total_loss, mel_loss1, mel_loss2, d_loss, p_loss, e_loss

def save_loss_plots(train_hist, val_hist, results_path, lang):
    os.makedirs(results_path, exist_ok=True)
    epochs = list(range(1, len(train_hist["total"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist["total"], label="Total Loss", linewidth=2.2, color="#1f77b4")
    plt.plot(epochs, train_hist["mel"], label="Mel + PostNet L1 Loss", linestyle="--", color="#ff7f0e")
    plt.plot(epochs, train_hist["var"], label="Variance Adaptor MSE", linestyle=":", color="#2ca02c")
    plt.title(f"FastSpeech 2 Training Loss Curve ({lang})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "training_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_hist["total"], label="Validation Total Loss", linewidth=2.2, color="#d62728")
    plt.plot(epochs, val_hist["mel"], label="Validation Mel + PostNet L1 Loss", linestyle="--", color="#9467bd")
    plt.plot(epochs, val_hist["var"], label="Validation Variance Adaptor MSE", linestyle=":", color="#8c564b")
    plt.title(f"FastSpeech 2 Validation Loss Curve ({lang})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "validation_loss.png"), dpi=200)
    plt.close()

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lang = config["language"]
    prep_dir = config["path"]["preprocessed_path"]
    ckpt_dir = config["path"]["ckpt_path"]
    res_dir = config["path"]["results_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(prep_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(prep_dir, "speakers.json"), "r") as f:
        speakers = json.load(f)
    with open(os.path.join(prep_dir, "stats.json"), "r") as f:
        stats = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training FastSpeech 2 for {lang} on {device} (Vocab={len(vocab)}, Speakers={len(speakers)})")

    train_ds = FastSpeech2Dataset(prep_dir, "train", vocab, speakers)
    val_ds = FastSpeech2Dataset(prep_dir, "val", vocab, speakers)

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = FastSpeech2(config, len(vocab), len(speakers), stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"], eta_min=5e-5)

    train_hist = {"total": [], "mel": [], "var": []}
    val_hist = {"total": [], "mel": [], "var": []}
    best_val = 999999.0
    epochs = config["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        t_tot, t_mel, t_var, n_batches = 0.0, 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"{lang} Epoch {epoch}/{epochs}", leave=False):
            texts, spks, pitches, energies, durations, mels, src_mask, mel_mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm, _, _, _, _ = model(
                texts, spks, src_mask, mel_mask, pitches, energies, durations
            )
            loss, m1, m2, dl, pl, el = compute_losses(
                mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm,
                mels, durations, pitches, energies, src_mask, mel_mask,
                model.p_mean, model.p_std, model.e_mean, model.e_std
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_thresh"])
            optimizer.step()

            t_tot += float(loss.item())
            t_mel += float((m1 + m2).item())
            t_var += float((dl + pl + el).item())
            n_batches += 1

        scheduler.step()
        train_hist["total"].append(t_tot / max(1, n_batches))
        train_hist["mel"].append(t_mel / max(1, n_batches))
        train_hist["var"].append(t_var / max(1, n_batches))

        model.eval()
        v_tot, v_mel, v_var, v_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                texts, spks, pitches, energies, durations, mels, src_mask, mel_mask = [b.to(device) for b in batch]
                mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm, _, _, _, _ = model(
                    texts, spks, src_mask, mel_mask, pitches, energies, durations
                )
                loss, m1, m2, dl, pl, el = compute_losses(
                    mel_out, mel_post, log_d_pred, p_pred_norm, e_pred_norm,
                    mels, durations, pitches, energies, src_mask, mel_mask,
                    model.p_mean, model.p_std, model.e_mean, model.e_std
                )
                v_tot += float(loss.item())
                v_mel += float((m1 + m2).item())
                v_var += float((dl + pl + el).item())
                v_batches += 1

        val_loss = v_tot / max(1, v_batches)
        val_hist["total"].append(val_loss)
        val_hist["mel"].append(v_mel / max(1, v_batches))
        val_hist["var"].append(v_var / max(1, v_batches))

        print(f"📈 [{lang}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_hist['total'][-1]:.4f} | Val Loss: {val_loss:.4f}")
        save_loss_plots(train_hist, val_hist, res_dir, lang)

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "speakers": speakers,
            "stats": stats,
            "config": config,
            "train_hist": train_hist,
            "val_hist": val_hist,
        }
        torch.save(ckpt_payload, os.path.join(ckpt_dir, "fastspeech2_latest.pt"))
        if max(0.0, best_val - val_loss) != 0.0:
            best_val = val_loss
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "fastspeech2_best.pt"))

    with open(os.path.join(res_dir, "loss_history.json"), "w") as f:
        json.dump({"train": train_hist, "val": val_hist}, f, indent=2)
    print(f"✅ Finished training {lang}! Best Val Loss: {best_val:.4f}")

if __name__ == "__main__":
    cfg_file = sys.argv[1] if max(0, len(sys.argv) - 1) != 0 else "config.yaml"
    main(cfg_file)
