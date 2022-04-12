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

深度学习的最新发展和大规模数据集的可用性导致强大的深度生成模型可以生成高度逼真的合成视频，同時最先进的生成模型具有大量有利的应用，但生成模型也被用于恶意目的。生成模型的此类应用之一則是深度伪造视频生成，其模型已经发展到难以区分真假视频的程度。此外 Deepfake 可用于不道德和恶意目的，例如传播虚假宣传、冒充政治领导人说或做不道德的事情，以及诽谤无辜的个人。該領域可以分为四类：人脸替换、人脸再现、人脸编辑和完整的人脸合成。另外要注意的地方在於 Deepfake 生成技术呈指数级增长，并且变得越来越难以检测，当前的检测系统不能有效地检测被操纵的介质。而在 Deepfake 检测挑战赛 (DFDC) 中，模型在未见过的数据上进行测试时的性能比在 DFDC 测试集上的性能差得多。

重要的地方在於泛化能力是现有深度伪造检测系统的主要关注点之一，各种各样的检测系统使用 CNN 和循环网络来检测被操纵的媒体。Li et al. 使用 CNN 来检测来自 Deepfake 数据集的图像中的面部扭曲伪影，其所提出的方法在存在可见的面部扭曲伪影的情况下效果很好。另外大多数 Deepfake 生成技术采用后处理程序来去除翘曲伪影，这使得检测 Deepfake 视频变得更加困难。现有方法的另一个限制是，大多数提出的系统对视频中的帧进行预测并对预测进行平均以获得整个视频的最终预测分数。所以它没有考虑框架之间的关系。而为了克服这个问题，該研究者提出了一种新颖的视频转换器来提取具有时间信息的空间特征。

Vaswani et al. 在  Attention is All you Need. 中首次提出了 Transformer 用于自然语言处理任务。自那时起，Transformer 在自然语言处理任务中表现出强大的性能，例如机器翻译、文本分类、问答和自然语言理解，而广泛使用的变压器架构包括来自变压器的双向编码器表示 (BERT)、鲁棒优化的 BERT 预训练 (RoBERTa)、生成式预训练变压器 (GPT) v1-v3。Transformer 模型可以自然地容纳用于特征学习的视频序列，为了提取更多信息特征，研究者在对齐的面部图像及其相应的 UV 纹理图上训练我们的模型。

现有方法使用对齐的 2D 人脸图像，这样的对齐方式只是将人脸居中，而不考虑人脸是否正面。当人脸未正面化时，相机未捕捉到的面部部分会导致面部信息丢失，并与正面化的人脸图像错位。同時使用 UV 纹理，所有面部图像都对齐到从生成的 3D 面部创建的 UV 贴图中。同時由于生成的 3D 人脸覆盖了所有的人脸部位，因此没有信息丢失，在 UV 贴图中，所有面部的面部部分可以位于相同的空间空间中。比如所有的鼻子部分都位于 UV 贴图上的同一区域，所以 UV 贴图中的面可以更好地对齐，另外为了处理人脸图像和 UV 纹理图的输入组合，我们在输入数据结构中使用可学习的片段嵌入。而段嵌入有助于模型区分同一数据结构中不同类型的输入。此外，該研究的研究者使用增量学习策略在不同数据集上逐步调整此研究的模型，以在新数据集上实现最先进的性能，同时保持先前数据集上的性能。

其贡献可以概括为三方面：其一，提出了一种带有人脸 UV 纹理图的视频转换器，用于深度伪造检测。在 ve 不同的公共数据集上的实验结果表明，此方法比最先进的方法实现了更好的性能。其二，所提出的段嵌入使网络能够提取更多信息特征，从而提高检测精度。其三，所提出的增量学习策略提高了所提出模型的泛化能力。在综合实验表明，此研究的模型可以在新数据集上取得良好的性能，同时保持其在以前数据集上的性能。

Figure 1: The architecture of the proposed video transformer, including the cropped face images and their corresponding UV texture maps as input, XceptionNet as backbone for image feature extraction and 12 transformer blocks for feature learning.

Figure 2: Illustration of the proposed incremental learning strategy. 

D1 represents the real data used to train the models.

Whereas, D2 comprises of FaceSwap and Deepfakes datasets.

D3 represents the Face2Face dataset and D4 represents Neural Textures dataset.

D5 and D6 represents DFDC dataset and D7 represents DeepFake Detection (DFD) dataset.

研究中所提出的视频转换器的架构，包括作为输入的裁剪后的人脸图像及其对应的 UV 纹理图，作为图像特征提取骨干的 XceptionNet 和用于特征学习的 12 个转换器块。

該研究所建议的增量学习策略的说明，當中的 D1 代表用于训练模型的真实数据，而 D2 由 FaceSwap 和 Deepfakes 数据集组成，此外 D3 代表 Face2Face 数据集，D4 代表神经纹理数据集。而 D5 和 D6 代表 DFDC 数据集，最後 D7 代表 DeepFake Detection (DFD) 数据集。