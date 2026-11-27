"""Jalur Transfer Learning (TL) - Spektrogram IMU ke MobileNetV2 Pralatih.

    python models/123450095/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"
JUMLAH_KELAS = 8  # 8 gestur dinamis sesuai rencana/t20-gestur-imu.yaml


def imu_ke_spektrogram_tensor(X: np.ndarray, ukuran_img: int = 64):
    """Konversi sinyal deret waktu IMU menjadi citra spektrogram 3-kanal (N, 3, H, W)."""
    import torch
    import torch.nn.functional as F

    arr = np.asarray(X, dtype=np.float32)
    # Bentuk standar IMU 100 Hz x 3 detik = 300 langkah waktu x 6 kanal (acc xyz + gyro xyz)
    if arr.ndim == 2:
        if arr.shape[1] % 6 == 0 and arr.shape[1] >= 60:
            arr = arr.reshape(len(arr), -1, 6)
        else:
            # Adaptasi untuk data simulasi 1D berdimensi kecil (mis. (N, 16))
            arr = arr[:, :, None]

    # Bentuk menjadi (N, T, C)
    if arr.ndim == 3 and arr.shape[1] < arr.shape[2]:
        arr = np.transpose(arr, (0, 2, 1))

    N, T, C = arr.shape
    t_tensor = torch.from_numpy(arr)  # (N, T, C)

    # Pilih maksimal 3 sumbu representatif untuk dipetakan ke 3 kanal RGB MobileNet
    if C >= 3:
        sinyal_3ch = t_tensor[:, :, :3]
    else:
        sinyal_3ch = t_tensor.repeat(1, 1, 3)[:, :, :3]

    # Hitung STFT (spektrogram) per kanal
    n_fft = min(64, max(8, T))
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft)

    kanal_spec = []
    for c in range(3):
        stft = torch.stft(
            sinyal_3ch[:, :, c],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        mag = torch.log1p(torch.abs(stft))  # log-spektrogram
        kanal_spec.append(mag)

    spec = torch.stack(kanal_spec, dim=1)  # (N, 3, F, T_frames)
    # Resize ke dimensi spasial 2D untuk masukan MobileNetV2
    spec_resized = F.interpolate(spec, size=(ukuran_img, ukuran_img), mode="bilinear", align_corners=False)

    # Normalisasi per sampel agar stabil saat masuk backbone pralatih
    mean = spec_resized.mean(dim=(2, 3), keepdim=True)
    std = spec_resized.std(dim=(2, 3), keepdim=True) + 1e-6
    return (spec_resized - mean) / std


def bangun_model_tl(jumlah_kelas: int):
    """Bangun MobileNetV2 pralatih dengan kepala klasifikasi 8 kelas gestur."""
    import torch.nn as nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except Exception:
        # Fallback tanpa unduh bobot jika lingkungan offline
        model = mobilenet_v2(weights=None)

    # Bekukan ekstraktor fitur pralatih (transfer learning feature extraction)
    for param in model.features.parameters():
        param.requires_grad = False

    in_fitur = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_fitur, jumlah_kelas),
    )
    return model


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Konversi IMU -> Spektrogram, latih MobileNetV2 pralatih, dan kembalikan prediksi."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    X_train = imu_ke_spektrogram_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    X_val = imu_ke_spektrogram_tensor(Xva)

    num_classes = max(JUMLAH_KELAS, int(y_train.max().item()) + 1)
    model = bangun_model_tl(num_classes).to(device)

    # AdamW dengan lr=1e-3 dipilih untuk fine-tuning kepala klasifikasi MobileNetV2
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size = 32
    epochs = 5
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val.to(device)).argmax(1).cpu().numpy()
        if final and Xte is not None:
            X_test = imu_ke_spektrogram_tensor(Xte)
            pred_test = model(X_test.to(device)).argmax(1).cpu().numpy()
        else:
            pred_test = None

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
