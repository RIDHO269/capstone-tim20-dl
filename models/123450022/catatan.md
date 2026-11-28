# Catatan Model 123450022 (Jalur RNN — Bidirectional GRU)

- **Jalur:** `rnn`
- **Arsitektur:** `nn.GRU` 2 lapis *bidirectional* (`hidden_size=64`) dengan penggabungan *mean pooling* dan *max pooling* pada dimensi waktu, diikuti `LayerNorm`, `Dropout(0.25)`, dan `Linear` menuju kelas gestur dinamis sesuai `rencana/t20-gestur-imu.yaml`.
- **Masukan:** Mendukung data simulasi sementara `(N, 16)` maupun sinyal IMU ponsel 100 Hz berdurasi 3 detik (`300 x 6` atau `200 x 6` setelah membuang 1 detik awal), dengan normalisasi per kanal yang dihitung hanya dari data latih.

## Alasan optimizer
`AdamW` (`lr=1e-3`, `weight_decay=1e-4`) dipilih karena adaptif terhadap gradien sekuens berulang (GRU) sekaligus memisahkan *weight decay* untuk menjaga generalisasi pada ukuran dataset terbatas.

## Regularisasi
- *Dropout* antar lapis GRU (`0.2`) dan pada kepala klasifikasi (`0.25`).
- `weight_decay=1e-4` pada `AdamW` dan `LayerNorm` sebelum lapisan klasifikasi.
- *Gradient clipping* (`max_norm=1.0`) untuk mencegah *exploding gradients*.

## Pencarian hyperparameter (tabel)
*(Diisi pada Sprint 2 saat pelatihan dan tuning pada data primer meja).*

## Ablation (>= 2)
*(Diisi pada Sprint 2: mis. GRU vs LSTM, unidir vs bidir, atau pooling rata-rata vs akhir).*

## Hasil 3 seed konfigurasi akhir
*(Diisi pada Sprint 2 setelah protokol data primer dibekukan).*
