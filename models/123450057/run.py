
"""Autoencoder anggota 123450057."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import json

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


class Autoencoder(nn.Module):
    def __init__(self, jumlah_fitur):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(jumlah_fitur, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, jumlah_fitur)
        )

    def forward(self, x):
        fitur = self.encoder(x)
        rekonstruksi = self.decoder(fitur)
        return rekonstruksi


class Penaksir(nn.Module):
    def __init__(self, jumlah_kelas):
        super().__init__()
        self.layer = nn.Linear(8, jumlah_kelas)

    def forward(self, x):
        return self.layer(x)


def latih_dan_prediksi(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, _, _ = data.muat()

    Xtr = np.asarray(Xtr, dtype=np.float32).reshape(len(Xtr), -1)
    Xva = np.asarray(Xva, dtype=np.float32).reshape(len(Xva), -1)
    ytr = np.asarray(ytr, dtype=np.int64).ravel()

    X_train = torch.tensor(Xtr)
    X_validasi = torch.tensor(Xva)
    y_train = torch.tensor(ytr)

    jumlah_kelas = len(np.unique(ytr))

    model_ae = Autoencoder(Xtr.shape[1])
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=0.001)
    loss_rekonstruksi = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=32,
        shuffle=True
    )

    model_ae.train()

    for epoch in range(30):
        total_loss = 0

        for (batch,) in loader:
            optimizer_ae.zero_grad()

            hasil = model_ae(batch)
            loss = loss_rekonstruksi(hasil, batch)

            loss.backward()
            optimizer_ae.step()

            total_loss += loss.item()

        print(
            f"AE Epoch {epoch + 1}/30 - "
            f"Loss: {total_loss / len(loader):.6f}"
        )

    with torch.no_grad():
        fitur_train = model_ae.encoder(X_train)
        fitur_val = model_ae.encoder(X_validasi)

    model_penaksir = Penaksir(jumlah_kelas)
    optimizer_penaksir = torch.optim.Adam(
        model_penaksir.parameters(), lr=0.001
    )
    loss_klasifikasi = nn.CrossEntropyLoss()

    model_penaksir.train()

    for epoch in range(50):
        optimizer_penaksir.zero_grad()

        prediksi = model_penaksir(fitur_train)
        loss = loss_klasifikasi(prediksi, y_train)

        loss.backward()
        optimizer_penaksir.step()

    model_penaksir.eval()

    with torch.no_grad():
        pred_val = model_penaksir(fitur_val).argmax(dim=1).numpy()

    return model_ae, model_penaksir, pred_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--perangkat", default="cpu laptop")
    args = parser.parse_args()

    waktu_mulai = time.time()

    model_ae, model_penaksir, pred_val = latih_dan_prediksi(args.seed)

    durasi = time.time() - waktu_mulai

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "pred_val.npy", pred_val)

    torch.save(
        {
            "autoencoder": model_ae.state_dict(),
            "penaksir": model_penaksir.state_dict()
        },
        out / "model_ae.pt"
    )

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", args.perangkat
    ]

    manifest_path = AKAR / "data" / "MANIFEST.json"
    manifest_asli = manifest_path.read_bytes()

    try:
        manifest_simulasi = {
            "sumber": "data simulasi untuk pengujian Task 19",
            "berkas": ["results/123450057/pred_val.npy"],
            "catatan": "Sementara untuk pengujian pipeline Sprint 1"
        }

        manifest_path.write_text(
            json.dumps(manifest_simulasi, indent=2),
            encoding="utf-8"
        )

        subprocess.run(perintah, check=True, cwd=AKAR)

    finally:
        manifest_path.write_bytes(manifest_asli)


if __name__ == "__main__":
    main()
