# Catatan Model — 123450095 (Jalur Transfer Learning / TL)

- **Jalur:** `tl` (Transfer Learning)
- **Arsitektur:** Spektrogram STFT 3-kanal $\rightarrow$ **MobileNetV2 pralatih** (`ImageNet` weights) dengan lapisan ekstraktor fitur dibekukan (*frozen backbone*) dan lapisan klasifikasi disesuaikan menjadi **8 kelas gestur dinamis** sesuai `rencana/t20-gestur-imu.yaml`.
- **Masukan:** Sinyal IMU ponsel 100 Hz selama 3 detik (`300 x 6`), ditransformasikan menjadi log-spektrogram 2D via `torch.stft` dan di-resize ke citra 3-kanal.
- **Optimizer:** `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) untuk mempercepat konvergensi saat melatih lapisan klasifikasi (*classifier head*) di atas fitur pralatih.
