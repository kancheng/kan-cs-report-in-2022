# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Video Transformer for Deepfake Detection with Incremental Learning

Khan, S. A., & Dai, H. (2021, October). Video Transformer for Deepfake Detection with Incremental Learning. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 1821-1828).

https://dl.acm.org/doi/abs/10.1145/3474085.3475332?sid=SCITRUS

https://arxiv.org/abs/2108.05307

```
Face forgery by deepfake is widely spread over the internet and this raises severe societal concerns. 

In this paper, we propose a novel video transformer with incremental learning for detecting deepfake videos. 

To better align the input face images, we use a 3D face reconstruction method to generate UV texture from a single input face image. 

The aligned face image can also provide pose, eyes blink and mouth movement information that cannot be perceived in the UV texture image, so we use both face images and their UV texture maps to extract the image features. 

We present an incremental learning strategy to fine-tune the proposed model on a smaller amount of data and achieve better deepfake detection performance. 

The comprehensive experiments on various public deepfake datasets demonstrate that the proposed video transformer model with incremental learning achieves state-of-the-art performance in the deepfake video detection task with enhanced feature learning from the sequenced data.
```

Deepfake 的面部偽造在互聯網上廣泛傳播，這引起了嚴重的社會擔憂，在該研究中，研究者提出了一種具有增量學習功能的新型影像轉換器，用於檢測深度偽造影像，而為了更好地對齊輸入人臉圖像，我們使用 3D 人臉重建方法從單個輸入人臉圖像生成 UV 紋理。其對齊的人臉圖像還可以提供在 UV 紋理圖像中無法感知的姿勢、眨眼和嘴巴運動資訊，因此研究者使用人臉圖像及其 UV 紋理圖來提取圖像特徵。最後研究者提出了一種增量學習策略，可以在更少量的資料上微調所提出的模型，並實現更好的深度偽造檢測性能。對各種公共 deepfake 數據集的綜合實驗表明，所提出的具有增量學習的視頻轉換器模型通過對序列數據的增強特徵學習，在 deepfake 視頻檢測任務中實現了最先進的性能。

Bibliography

```
@inproceedings{khan2021video,
  title={Video Transformer for Deepfake Detection with Incremental Learning},
  author={Khan, Sohail Ahmed and Dai, Hang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1821--1828},
  year={2021}
}
```

## 1. INTRODUCTION

Recent developments in deep learning and the availability of large scale datasets have led to powerful deep generative models that can generate highly realistic synthetic videos. 

State-of-the-art generative models have enormous amount of advantageous applications, but the generative models are also used for malicious purposes.

One such application of the generative models is deepfake video generation.

Generative models have evolved to an extent that, it is diffcult to classify the real and the fake videos.

Deepfake can be used for unethical and malicious purposes, for example, spreading false propaganda, impersonating political leaders saying or doing unethical things, and defaming innocent individuals. 

Deepfake can be grouped into four categories: face replacement, facial re-enactment, face editing, and complete face synthesis.

Deepfake generation techniques are increasing exponentially and becoming more and more diffcult to detect. 

Current detection systems are not in a capacity to detect manipulated media effectively. 

In Deepfake Detection Challenge (DFDC), the models achieve much worse performance when tested on unseen data than that on the DFDC test set. 

Generalization capability is one of the major concerns in the existing deepfake detection systems. 

A wide variety of detection systems employ CNNs and recurrent networks to detect manipulated media. 

Li et al. employ CNNs to detect face warping artifacts in images from the deepfake datasets. 

The proposed approach works well in cases where there are visible face warping artifacts.

Most of the deepfake generation techniques employ post-processing procedures to remove the warping artifacts, which makes it more di"cult to detect deepfake videos. 

Another limitation of the existing approaches is that, most of the proposed systems make predictions on the frames in a video and average the predictions in order to get a !nal prediction score for the whole video.

So it fails to consider the relationships among frames. 

To overcome this, we propose a novel video transformer to extract spatial features with the temporal information.

Transformers were first proposed for natural language processing tasks, by Vaswani et al., in [50]. 

Since then, transformers show powerful performance in the natural language processing tasks, for example, machine translation, text classification, question-answering, and natural language understanding. 

The widely used transformer architectures include Bidirectional Encoder Representations from Transformers (BERT), Robustly Optimized BERT Pre-training (RoBERTa), Generative Pre-trained Transformer (GPT) v1-v3. 

The transformer models can naturally accommodate the video sequences for the feature learning.

To extract more informative features, we train our models on the aligned facial images and their corresponding UV texture maps. 

The existing methods use aligned 2D face images.

Such an alignment only centralizes the face without considering whether the face is frontalized. 

When the face is not frontalized, the face part that is not captured by the camera can cause facial information loss
and misalignment with the face images that are frontalized. 

With the UV texture, all face images are aligned into the UV map that is created from the generated 3D faces.

Since the generated 3D faces cover all the facial parts, there is no information loss.

In UV map, the facial part for all the faces can be located in the same spatial space.

For example, all the nose parts are located in the same region on the UV map.

So the faces in UV maps are better aligned.

To deal with the input combination of face image and UV texture map, we use learnable segment embeddings in the input data structure.

The segment embeddings help the model to distinguish different types of inputs in the same data structure.

Furthermore, we use an incremental learning strategy for !netuning our models on different datasets incrementally to achieve state-of-the-art performance on new datasets while maintaining the performance on the previous datasets.

Our contributions can be summarized in three-fold:

- We propose a video transformer with face UV texture map for deepfake detection. The experimental results on !ve
different public datasets show that our method achieves better performance than state-of-the-art methods.

- The proposed segment embedding enables the network to extract more informative features, thereby improving the
detection accuracy.

- The proposed incremental learning strategy improves the generalization capability of the proposed model. The comprehensive experiments show that our model can achieve good performance on a new dataset, while maintaining their performance on previous dataset.