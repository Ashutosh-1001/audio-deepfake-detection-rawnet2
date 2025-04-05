# audio-deepfake-detection-rawnet2


Part 1: Research & Model Selection

✅ 1. RawNet2
Key Technical Innovation: An end-to-end deep neural network that processes raw audio waveforms directly, eliminating the need for manual feature extraction.

Performance: Achieves an impressive ~0.29% Equal Error Rate (EER) on the ASVspoof 2019 LA dataset.

Why it’s promising: Offers a streamlined pipeline and potential for real-time inference, while delivering strong detection performance.

Challenges: Demands high GPU memory and is prone to overfitting, especially on limited datasets.

✅ 2. AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal modeling)
Key Technical Innovation: Integrates raw waveform and spectrogram features through a combination of attention mechanisms and graph-based modeling.

Performance: Achieves <1% EER on ASVspoof 2021, making it one of the top-performing models.

Why it’s promising: Demonstrates exceptional robustness to a wide range of spoofing attacks and generalizes well across datasets.

Challenges: The architecture is complex and computationally heavy, leading to slower inference times.

✅ 3. LFCC + GMM (Linear Frequency Cepstral Coefficients + Gaussian Mixture Model)
Key Technical Innovation: A classical approach using handcrafted features and statistical modeling, with no reliance on deep learning.

Performance: Around 10% EER on benchmark datasets.

Why it’s promising: Offers very fast inference, low computational overhead, and is easy to interpret and implement.

Challenges: Struggles significantly with generalization, especially when exposed to unseen spoofing techniques.


