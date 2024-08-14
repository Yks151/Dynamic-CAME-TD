This study proposed a performance-interpretable deep learning model for rolling bearing fault diagnosis that integrates an intelligent fusion of sound and vibration signals and self-supervised learning via an interpretable attention mechanism. A deep learning decoder framework with a compressed attention mechanism encoder (CAME) is developed to automatically learn the correlation between sound and vibration signals and the fusion method, eliminating the need for manual feature extraction and multi-model construction. By introducing the dynamic attention mechanism, the strength of the correlation between sound and vibration signals can be sensed in real-time to adapt to different scenarios flexibly. When the correlation is strong, due to the high similarity between the signals, a complex feature weight fusion strategy is employed to extract and fuse the essential features of different modalities more efficiently, enabling this fusion to mutually enhance the expressive power of the features for feature fusion. Whereas, when the correlation is weak, the correlation between the signals is low and forcing a complex fusion may introduce more noise and redundant information, therefore a hybrid input strategy is used. The CAME-TD (CAME-Transformer Decoder) model dynamically updates the correlation thresholds and fusion strategies using regularized loss constraints to ensure adaptation to multimodal signal differences. During model training, visual analysis of the attention mechanism role weights and feature learning helps in parameter optimization and performance evaluation. The experimental results demonstrate the effectiveness of the proposed methodology, with an improvement in fault diagnosis performance under various operating and noise conditions compared to a single signal input. Moreover, the CAME-TD model not only achieves considerable diagnostic performance but also enhances interpretability, providing a new approach for rolling bearing fault diagnosis.