# 🎙️ audio-deepfake-detection-rawnet2

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Torch](https://img.shields.io/badge/PyTorch-1.13-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Prototype-yellow?style=flat-square)

---



---

## 🧭 Table of Contents

- [🔍 Research & Model Selection](#-part-1-research--model-selection)
- [⚙️ Implementation Documentation](#️-1-implementation-documentation)
- [📊 Analysis](#-2-analysis)
- [🪞 Reflection Questions](#-3-reflection-questions)



## 🔍 Part 1: Research & Model Selection

### ✅ 1. RawNet2
- **Key Technical Innovation**: An end-to-end deep neural network that processes raw audio waveforms directly, eliminating the need for manual feature extraction.
- **Performance**: Achieves an impressive ~0.29% Equal Error Rate (EER) on the ASVspoof 2019 LA dataset.
- **Why it’s promising**: Offers a streamlined pipeline and potential for real-time inference, while delivering strong detection performance.
- **Challenges**: Demands high GPU memory and is prone to overfitting, especially on limited datasets.

---

### ✅ 2. AASIST
- **Key Technical Innovation**: Integrates raw waveform and spectrogram features through a combination of attention mechanisms and graph-based modeling.
- **Performance**: Achieves <1% EER on ASVspoof 2021, making it one of the top-performing models.
- **Why it’s promising**: Demonstrates exceptional robustness to a wide range of spoofing attacks and generalizes well across datasets.
- **Challenges**: The architecture is complex and computationally heavy, leading to slower inference times.

---

### ✅ 3. LFCC + GMM
- **Key Technical Innovation**: A classical approach using handcrafted features and statistical modeling, with no reliance on deep learning.
- **Performance**: Around 10% EER on benchmark datasets.
- **Why it’s promising**: Offers very fast inference, low computational overhead, and is easy to interpret and implement.
- **Challenges**: Struggles significantly with generalization, especially when exposed to unseen spoofing techniques.

---

## ⚙️ Part 3: Documentation & Analysis

### 🛠️ 1. Implementation Documentation

#### 🧩 Challenges Encountered
- Handling the ASVspoof2019 dataset structure across train/test/metadata directories.
- Aligning audio files with their respective labels from protocol files.
- RawNet2 is heavy and difficult to train/infer without powerful resources or pretrained weights.

#### 🛠️ How These Were Addressed
- Used a 5-sample subset for testing and demo purposes.
- Built a dummy RawNet2-style model to validate preprocessing and prediction logic.
- Commented clearly where real model integration can be swapped in.

#### 🤔 Assumptions Made
- Protocol files are reliable for label mapping.
- Dummy model suffices for proof of concept.
- Small sample set is representative enough for testing pipeline logic.

---

### 📊 2. Analysis

#### 🧠 Why This Model Was Selected
RawNet2 is a research benchmark model that directly works on raw waveforms — ideal for deepfake detection tasks. It removes manual feature engineering and simplifies the end-to-end pipeline.

#### 🧬 How the Model Works (High-Level)
1. **Convolutional Layers** → Learn time-domain audio features  
2. **Residual Blocks** → Preserve important signals across layers  
3. **Pooling Layers** → Reduce dimensionality  
4. **Fully Connected Layer** → Outputs spoof/bona fide prediction

#### ✅ Strengths Observed
- No MFCC/spectrograms needed.
- Real-time compatible end-to-end structure.
- Code and label integration working as expected.

#### ❌ Weaknesses Observed
- Small dataset leads to poor generalization.
- No robustness testing against varied spoof types or noisy data.

#### 🚀 Suggestions for Future Improvement
- Add data augmentation: background noise, time stretch, pitch shift.
- Consider adversarial training or domain adaptation for robustness.

---

### 🪞 3. Reflection Questions

#### 1. What were the most significant challenges in implementing this model?
Setting up the ASVspoof dataset correctly and working with RawNet2's structure were the biggest hurdles. Resource limits meant simulating the core behavior instead of using the real model.

#### 2. How might this approach perform in real-world conditions vs. research datasets?
RawNet2 may underperform in noisy environments or on devices with different microphones. It’s crucial to fine-tune or retrain on real-world spoof samples.

#### 3. What additional data or resources would improve performance?
- Larger, diverse spoof datasets  
- Real-world audio recordings  
- Pretrained models or GPU compute

#### 4. How would you approach deploying this model in a production environment?
- Export model using ONNX or TorchScript  
- Use FastAPI + Torch backend  
- WebRTC for streaming voice input  
- Monitor predictions and retrain periodically


