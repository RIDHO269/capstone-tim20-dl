"""Jalur RNN (Bidirectional GRU) untuk klasifikasi 8 gestur dinamis IMU ponsel.

    python models/123450022/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name
JALUR = "rnn"
JUMLAH_KELAS_TARGET = 8  # 8 gestur dinamis sesuai rencana/t20-gestur-imu.yaml


def atur_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ke_tensor_imu(X: np.ndarray):
    """Ubah input simulasi (N, 16) maupun data IMU (N, T, 6) / (N, T*6) menjadi (N, T, C)."""
    import torch

    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[1] < arr.shape[2] and arr.shape[1] == 6:
            arr = np.transpose(arr, (0, 2, 1))
        return torch.from_numpy(arr)
    if arr.ndim == 2:
        if arr.shape[1] >= 60 and arr.shape[1] % 6 == 0:
            return torch.from_numpy(arr.reshape(arr.shape[0], arr.shape[1] // 6, 6))
        # Adaptasi untuk data simulasi 1D sementara (mis. (N, 16))
        return torch.from_numpy(arr.reshape(arr.shape[0], arr.shape[1], 1))
    return torch.from_numpy(arr.reshape(arr.shape[0], 1, 1))


def bangun_model_rnn(in_channels: int, num_classes: int, hidden_size: int = 64):
    import torch
    from torch import nn

    class PengklasifikasiGRU(nn.Module):
        """Bidirectional GRU dengan mean + max pooling untuk sekuens IMU."""

        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=in_channels,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True,
            )
            feat_dim = hidden_size * 4
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(p=0.25),
                nn.Linear(feat_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            mean_pool = out.mean(dim=1)
            max_pool = out.max(dim=1).values
            return self.head(torch.cat([mean_pool, max_pool], dim=1))

    return PengklasifikasiGRU()


def prediksi_batch(model, x, batch_size: int = 64) -> np.ndarray:
    import torch

    model.eval()
    hasil = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(x[i : i + batch_size])
            hasil.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(hasil, axis=0)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih Bidirectional GRU pada split latih dan kembalikan (pred_val, pred_test|None)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    atur_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_raw, ytr_raw, Xva_raw, _, Xte_raw, _ = data.muat()
    Xtr = ke_tensor_imu(Xtr_raw)
    Xva = ke_tensor_imu(Xva_raw)
    Xte = ke_tensor_imu(Xte_raw) if (final and Xte_raw is not None) else None

    # Normalisasi per kanal hanya dihitung dari data latih
    mean = Xtr.mean(dim=(0, 1), keepdim=True)
    std = Xtr.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    if Xte is not None:
        Xte = (Xte - mean) / std

    ytr = torch.as_tensor(np.asarray(ytr_raw), dtype=torch.long)
    num_classes = max(int(ytr.max().item()) + 1, 2)
    in_channels = Xtr.shape[2]

    model = bangun_model_rnn(in_channels=in_channels, num_classes=num_classes, hidden_size=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
    epochs = 25 if in_channels == 6 else 10

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    pred_val = prediksi_batch(model, Xva.to(device))
    pred_test = prediksi_batch(model, Xte.to(device)) if Xte is not None else None
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
