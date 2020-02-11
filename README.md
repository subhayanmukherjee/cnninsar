# InSAR Image Filtering & Coherence Estimation via CNN
First ever Convolutional Neural Network-based filtering and point-wise signal quality quantification methodology for InSAR. Interferometric Synthetic Aperture Radar (InSAR) imagery for estimating ground movement, based on microwaves reflected off ground targets is gaining increasing importance in remote sensing. However, noise corrupts microwave reflections received at satellite and contaminates the signal's wrapped phase. We show the effectiveness of autoencoder CNN architectures to learn InSAR image denoising filters in the absence of clean ground truth images, and for artefact reduction in estimated coherence through intelligent preprocessing of training data.

Please cite the below [paper](https://doi.org/10.1109/ICSENS.2018.8589920) if you use the code in its original or modified form:

*S. Mukherjee, A. Zimmer, N. K. Kottayil, X. Sun, P. Ghuman and I. Cheng, "CNN-Based InSAR Denoising and Coherence Metric," 2018 IEEE SENSORS, New Delhi, 2018, pp. 1-4.*

## Guidelines

1. The entire code for generating the training datasets and training the filtering and coherence estimation models is provided in [train_coh.py](https://github.com/subhayanmukherjee/cnninsar/blob/master/train_coh.py).
2. If you want to test the trained model on a simulated images dataset, you can use [buildset_noisy_sim.py](https://github.com/subhayanmukherjee/cnninsar/blob/master/buildset_noisy_sim.py), but you need to first download download the [simulator](https://github.com/Lucklyric/InSAR-Simulator) (main branch) and place it in the same folder as all other codes.
