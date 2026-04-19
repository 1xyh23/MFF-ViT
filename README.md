# MFF-ViT: Localizing Forged Targets in Remote Sensing Imagery Using Spectral Inconsistency Cues

This repository contains the official implementation of the paper **"Localizing Forged Targets in Remote Sensing Imagery Using Spectral Inconsistency Cues to Support Authenticity Assessment"**.

We propose **MFF-ViT**, a frequency-aware vision transformer that leverages spectral inconsistencies between manipulated regions and their surrounding context for accurate and interpretable forged-target localization in satellite imagery. Our method achieves state-of-the-art performance across three remote sensing forgery benchmarks, with particularly robust behavior on AIGC-based removals and small targets.

---

## 🛠️ Experimental Framework
All experiments are conducted using the **IMDL-BenCo** comprehensive benchmark and codebase for image manipulation detection and localization.

- **IMDL-BenCo Official Repository**: [https://github.com/scu-zjz/IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo)
- For methods with available code, we reproduce results under the IMDL-BenCo framework using a unified training protocol.

---

## 📊 Datasets
We evaluate our method on three representative remote sensing forgery benchmarks:

| Dataset | Description | Source |
|---------|-------------|--------|
| **RSTFD** | Target-oriented forgery dataset covering airports, ports, and open-water scenes, with unified support for splicing, copy-move, and AIGC-based removal | This work |
| **SMTD** | High-resolution satellite map tampering dataset focusing on splicing forgery in urban scenes | [Spectral information guidance network for tampering localization of high-resolution satellite map](https://www.sciencedirect.com/science/article/pii/S0957417424021528)<br>Dataset link: [https://pan.baidu.com/s/12Gc8atDdHl8iYwwz_EwgaQ](https://pan.baidu.com/s/12Gc8atDdHl8iYwwz_EwgaQ) |
| **Fake-Vaihingen** | Diffusion-based AIGC forgery dataset built on the Vaihingen aerial imagery benchmark | [FLDCF: A Collaborative Framework for Forgery Localization and Detection in Satellite Imagery](https://ieeexplore.ieee.org/document/10586763)<br>Dataset link: [https://github.com/littlebeen/Forgery-localization-for-remote-sensing](https://github.com/littlebeen/Forgery-localization-for-remote-sensing) |

---

## 📈 Experimental Results

### Overall Performance Across Three Datasets
MFF-ViT achieves the best average performance across all three benchmarks, outperforming all state-of-the-art methods in both localization accuracy and robustness.

| Methods | Venue/Year | SMTD |  | Fake-Vaihingen |  | RSTFD |  | Average |  | Complexity (RSTFD) |  |
|---------|------------|------|-----|----------------|-----|-------|-----|---------|-----|---------------------|-----|
|  |  | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **Params (M) ↓** | **FLOPs (T) ↓** |
| CAT-NET | WACV/2021 | 0.9681 | 0.9387 | 0.7811 | 0.6582 | 0.8613 | 0.7899 | 0.8702 | 0.7956 | 116.74 | 0.134 |
| PSCC | TCSVT/2022 | -- | -- | 0.7862 | 0.7022 | 0.7756 | 0.6785 | 0.7809 | 0.6904 | **3.67** | 0.368 |
| TruFor | CVPR/2023 | 0.9768 | 0.9549 | 0.8330 | 0.7262 | 0.9033 | 0.8398 | 0.9044 | 0.8403 | 68.70 | 0.231 |
| IML-ViT | NeurIPS/2024 | 0.9765 | 0.9549 | 0.9188 | 0.8535 | 0.8740 | 0.8048 | 0.9231 | 0.8711 | 91.75 | 0.121 |
| M-FLNet | TGRS/2024 | 0.9703 | 0.9430 | 0.9167 | 0.8462 | 0.7512 | 0.6755 | 0.8794 | 0.8216 | 31.27 | 0.735 |
| Mesorch | AAAI/2025 | 0.9734 | 0.9485 | 0.8758 | 0.7866 | 0.8807 | 0.8025 | 0.9100 | 0.8459 | 85.75 | 0.122 |
| **MFF-ViT (Ours)** | - | **0.9779** | **0.9572** | **0.9262** | **0.8659** | **0.9134** | **0.8531** | **0.9392** | **0.8921** | 92.54 | 0.130 |

### Fine-grained Performance on RSTFD
MFF-ViT shows significant advantages in challenging scenarios, especially AIGC-based object removal and small-target forgery.

| Methods | Type-Subset |  |  |  |  |  | Scale-Subset |  |  |  |  |  |
|---------|-------------|-----|-------------|-----|-------------|-----|-------------|-----|-------------|-----|-------------|-----|
|  | **Splicing** |  | **Copy-Move** |  | **AIGC Removal** |  | **Small (<1%)** |  | **Medium (1-3%)** |  | **Large (>3%)** |  |
|  | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** | **F1 ↑** | **IoU ↑** |
| CAT-NET | 0.9209 | 0.8624 | 0.9186 | 0.8603 | 0.7341 | 0.6366 | 0.8030 | 0.7048 | 0.8769 | 0.8197 | 0.9571 | 0.9261 |
| PSCC | 0.8732 | 0.7872 | 0.8737 | 0.7888 | 0.5785 | 0.4577 | 0.6812 | 0.5589 | 0.8165 | 0.7230 | 0.9265 | 0.8745 |
| TruFor | <u>0.9403</u> | <u>0.8941</u> | **0.9424** | <u>0.8950</u> | <u>0.8263</u> | <u>0.7289</u> | <u>0.8557</u> | <u>0.7680</u> | **0.9370** | <u>0.8856</u> | <u>0.9699</u> | <u>0.9435</u> |
| IML-ViT | 0.9366 | 0.8888 | 0.9311 | 0.8799 | 0.7533 | 0.6444 | 0.8331 | 0.7382 | 0.8971 | 0.8398 | 0.9353 | 0.9068 |
| M-FLNet | 0.8918 | 0.8232 | 0.8853 | 0.8148 | 0.4745 | 0.3861 | 0.6666 | 0.5737 | 0.7791 | 0.7108 | 0.8925 | 0.8441 |
| Mesorch | 0.9130 | 0.8496 | 0.9160 | 0.8530 | 0.8125 | 0.7043 | 0.8257 | 0.7179 | 0.9166 | 0.8544 | 0.9599 | 0.9270 |
| **MFF-ViT (Ours)** | **0.9472** | **0.9059** | <u>0.9423</u> | **0.8968** | **0.8502** | **0.7558** | **0.8761** | **0.7914** | <u>0.9318</u> | **0.8861** | **0.9713** | **0.9471** |

---

## 🧠 Network Architecture
The implementation of MFF-ViT will be released upon acceptance of the paper.

---


---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
