# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection

Wang, J., Wu, Z., Chen, J., & Jiang, Y. G. (2021). M2tr: Multi-modal multi-scale transformers for deepfake detection. arXiv preprint arXiv:2104.09770.

https://arxiv.org/abs/2104.09770

```
The widespread dissemination of forged images generated by Deepfake techniques has posed a serious threat to the trustworthiness of digital information. 

This demands effective approaches that can detect perceptually convincing Deepfakes generated by advanced manipulation techniques. 

Most existing approaches combat Deepfakes with deep neural networks by mapping the input image to a binary prediction without capturing the consistency among different pixels. 

In this paper, we aim to capture the subtle manipulation artifacts at different scales for Deepfake detection. 

We achieve this with transformer models, which have recently demonstrated superior performance in modeling dependencies between pixels for a variety of recognition tasks in computer vision. 

In particular, we introduce a Multi-modal Multi-scale TRansformer (M2TR), which uses a multi-scale transformer that operates on patches of different sizes to detect the local inconsistency at different spatial levels. 

To improve the detection results and enhance the robustness of our method to image compression, M2TR also takes frequency information, which is further combined with RGB features using a cross modality fusion module. 

Developing and evaluating Deepfake detection methods requires large-scale datasets. However, we observe that samples in existing benchmarks contain severe artifacts and lack diversity. 

This motivates us to introduce a high-quality Deepfake dataset, SR-DF, which consists of 4,000 DeepFake videos generated by state-of-the-art face swapping and facial reenactment methods. 

On three Deepfake datasets, we conduct extensive experiments to verify the effectiveness of the proposed method, which outperforms state-of-the-art Deepfake detection methods.
```

Deepfake 技術所產生的偽造圖像廣泛傳播對數位資訊的可信度構成了嚴重威脅，這需要有效的方法來檢測由先進技術所生成具有感知力的 Deepfake 成果。大多數現有方法通過將輸入圖像對應到二進制預測而不捕獲不同像素之間的一致性來使用深度神經網絡來對抗 Deepfakes 技術。在該研究中，研究者旨在為 Deepfake 檢測捕獲不同尺度的細微操作偽影，並通過轉換器模型實現了這一點，該模型最近在為計算機視覺中的各種識別任務建模像素之間的依賴關係方面表現出卓越的性能。同時研究者介紹了一種多模態多尺度變換器（M2TR），它使用多尺度變換器對不同大小的補丁進行操作，以檢測不同空間級別的局部不一致性，為了改善檢測結果並增強我們方法對圖像壓縮的魯棒性，M2TR 還獲取頻率信息，並使用交叉模態融合模塊將其與 RGB 特徵進一步結合。開發和評估 Deepfake 檢測方法需要大規模的數據集。此研究觀察到現有基準中的樣本包含嚴重的偽影並且缺乏多樣性，這促使此研究引入了一個高品質的 Deepfake 資料集 SR-DF，它由 4,000 個由最先進的面部交換和面部重演方法去生成的 DeepFake 組成影像，最後在三個 Deepfake 資料集上，研究者進行了廣泛的實驗來驗證所提出方法的有效性，該方法優於最先進的 Deepfake 檢測方法。

Bibliography

```
@article{wang2021m2tr,
  title={M2tr: Multi-modal multi-scale transformers for deepfake detection},
  author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2104.09770},
  year={2021}
}
```

## Note

此外該研究提出了一種名為 M2TR 的多模態多尺度變壓器 ，首先運用 CNN 模型提取特徵，然後生成作為 Transformer 模型的輸入，用於捕捉不同區域在多尺度上的不一致性，並在此基礎上增加了頻域信息，融合後輸出結果。

0. DCT 部分

DCT 也就是離散餘弦變換的縮寫，將圖像從空域轉化到頻域，其研究者認為，圖像的低頻信息集中在 0-1/16 部分，中頻信息集中在 1/16-1/8，剩下的部分是圖像的高頻信息，通過構造三個濾波器來完成提取低中高頻信息。通過經過 DCT 變換後的圖像中黑色的部分而捨棄白色的部分，從而得到提取低頻，中頻，高頻信息。並用將得到的信息進行逆變換，將低中高頻信息組合在一起，作為頻域信息的輸入，送入卷積網絡中，將其輸出作為提取到的頻域特徵。


1. Multi-scale Transformer

希望定位篡改偽影與其他區域不一致，因此需要建模長期關係，計算相似度，引入多尺度的 Transformer，來覆蓋不同大小的區域。


輸入圖片再 backbone 提取 shallow feature，然後分成不同的大小去計算 patch-wise 的 self attention，也就是每個 Patch（rh*rh*c）展開成一維向量，使用 FC 層 embed 到 Query Embeddings，同樣得到 k 和 v，最後通過矩陣相乘得到相速度最後通過 Softmax 輸出。

$$
\alpha_{i, j}^{h}=\operatorname{softmax}\left(\frac{\boldsymbol{q}_{i}^{h} \cdot\left(\boldsymbol{k}_{j}^{h}\right)^{T}}{\sqrt{r_{h} \times r_{h} \times C}}\right), 1 \leq i, j \leq N
$$

最後相乘相加得到查詢 Patch 的輸出：

$$
\boldsymbol{o}_{i}^{h}=\sum_{j=1}^{N} \alpha_{i, j}^{h} v_{j}^{h}
$$

接收所有輸出 Sitch and Reshape 原本的 Resolution，不同的頭拼接通過 Residual Block 得到輸出結果。

$$
f_{m t} \in \mathbb{R}(H / 4) \times(W / 4) \times C
$$

這一部分與視覺注意力機制類似，唯一的區別是，這一部分使用多種不同尺度的 Patch 對圖像進行採樣。

舉個例子，假設使用的是分辨率為 56 * 56 的圖像，一般的視覺注意力機制會使用 14 * 14 的 Patch 對圖像進行採樣，採樣後將圖片分為 4 * 4 也就是 16 個塊，隨後再對這16個塊進行編碼，得到 16 個 token，最後使用自註意力機制，通過每個串得到 qkv，最後得到輸出結果。

而所謂的多尺度是將這一過程重複數次，每次使用的 Patch 大小不同，第一次我們使用與圖像相同大小的 Patch，採用後得到 1 個塊。第二次使用 28 * 28 大小的 Patch，得到 2 * 2 也就是 4 個塊。第三次使用 14 * 14 大小的 Patch，得到 4 * 4 = 16 個塊，最後使用 7 * 7 大小的 Patch，得到 8 * 8 = 64 個塊。每一個尺度都會再各自尺度分別利用自註意力機制進行計算，得到各自尺度的輸出結果，最後將各個輸出結合再一起，送入卷積中，得到 Transformer 部分的特徵提取。


另外再詳細說明單個多頭自註意力 (Mutil-head Self Attention, MSA) 的操作過程。

在 VIT 中，首先將圖片按照一定的尺寸進行分割，變成一個個 Token，舉例來說，當輸入圖片是 224 * 224， 而 Patch 大小為 16 * 16，就會得到 14 * 14 也就是 196 個塊。隨後將每個塊放入 embed 模塊，映射成 196 個 Token，每個 Token 的長度被規定為 768(14 * 14 * 3)。這時圖片從 [b, 3, 224, 224] 變成了 [b, 196, 768]。然後此時需要增加一個 cls 和一個位置信息 pos，所謂的 cls 是一個特殊字符，它的具體作用是用於分類，在 nlp 中 cls 通常位於第一位，位置信息表示每一個 Token 的位置，由於改變 Token 的位置會將整張圖片的語義信息改變，所以添加位置信息是必要的。 cls 的大小是 [b, 1, 768], 位置信息的大小是 [ b, 197, 768]，至於是用 197 的原因呢，是因為在原始 Token 的基礎上會將 cls 與 Token 進行 concatenate，從而使 Token 的大小變成了 [b, 196+1, 768]，然後將 Token 與位置信息相加，而不是 concatenate，這樣就完成了 embed 操作。 Embed 完成後，我們將 Token 送入到 ATTENTION 中，同時為了並行計算，此工作會直接使用 nn.Linear, 將輸出從 [b, 197, 768] 變成 [b, 197, 3*768]，然後 reshape 成 [3,b, 197, 768]，依次得到 qkv，然後按照 VIT 論文中的計算公式，得到輸出結果。此時要注意的地方在於，如果是有多個頭部，那麼得到的 qkv 應該是 [3,b, head_num, 197, 768/head_num]，所有的操作分成多個頭部進行，得到的輸出再拼接在一起，但由於此工作是並行計算，所以最後得到的輸出應該是 [b, head_num, 197, 768/head_num]，然後直接 reshape 變成 [b, 197, 768]就完成了全部操作。在該問題中，由於是使用 Transformer 來提取特徵圖，而非直接分類，所以並不需要使用 cls token，並且這樣也方便後續處理工作。


2. Cross Modality Fusion

CMF 模塊是一個特徵融合模塊，將上述三個模塊的輸出融合再一起。具體的操作與一個多頭自註意力機制類似，而與 MSA 相似地方在於，CMF 模塊中也存在 qkv ，其 q 是通過空域特徵圖卷積得到，k 和 v 是通過頻域特徵圖卷積得到。首先按照 MSA 算法將空域特徵圖和頻域特徵圖融合，具體的計算公式如下所示。

$$
\begin{aligned}
&Q=\operatorname{Conv}_{q}\left(f_{s}\right), K=\operatorname{Conv}_{k}\left(f_{f q}\right), V=\operatorname{Conv}_{v}\left(f_{f q}\right) \\
&f_{\text {fuse }}=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{H / 4 \times W / 4 \times C}}\right) V
\end{aligned}
$$

得到的融合特徵 $f_{fuse}$ 在和空域特徵和 Transformer 輸出的特徵圖相加，作為特徵圖送入卷積網絡中進一步提取特徵。其具體的計算公式如下。

$$
f_{c m f}=C o n v_{3 \times 3}\left(f_{s}+f_{m t}+f_{f u s e}\right)
$$


## 1. INTRODUCTION

Figure 1: Visual artifacts of forged images in existing datasets, including color mismatch (row 1 col 1, row 2 col 3,
row 3 col 1, row 3 col 2, row 3 col 3) , shape distortion (row 1 col 3, row 2 col 1), visible boundaries (row 2 col 2), and facial blurring (row 1 col 2, row 4 col 1, row 4 col 2, row 4 col3).

圖 1：現有數據集中偽造圖像的視覺偽影，包括顏色不匹配 (row 1 col 1, row 2 col 3,
row 3 col 1, row 3 col 2, row 3 col 3) 、形狀失真 (row 1 col 3, row 2 col 1)、可見邊界 (row 2 col 2) 和臉部模糊 (row 1 col 2, row 4 col 1, row 4 col 2, row 4 col3)。

Recent years have witnessed the rapid development of Deepfake techniques, which enable attackers to manipulate the facial area of an image and generate a forged image. 

As synthesized images are becoming more photo-realistic, it is extremely difficult to distinguish whether an image/video has been manipulated even for human eyes. 

At the same time, these forged images might be distributed on the Internet for malicious purposes, which could bring societal implications. 

The above challenges have driven the development of Deepfake forensics using deep neural networks.

Most existing approaches take as inputs a face region cropped out of an entire image and produce a binary real/fake prediction with deep CNN models. 

These methods capture artifacts from the face regions in a single scale with stacked convolutional operations. 

While decent detection results are achieved by stacked convolutions, they excel at modeling local information but fails
to consider the relationships of pixels globally due to constrained receptive field.

We posit that relationships among pixels are particularly useful for Deepfake detection, since pixels in certain artifacts are clearly different from the remaining pixels in the image. 

On the other hand, we observe that forgery patterns vary in sizes. 




近年來 Deepfake 技術迅速發展，使攻擊者能夠操縱圖像的面部區域並生成偽造圖像，且隨著合成圖像變得更加逼真，即使是人眼也很難區分圖像/視頻是否已被操縱。同時這些偽造的圖像可能會出於惡意目的在互聯網上傳播，這可能會帶來社會影響。上述挑戰推動了使用深度神經網絡的 Deepfake 取證技術的發展。

大多數現有方法將從整個圖像中裁剪出的人臉區域作為輸入，並使用深度 CNN 模型生成二進制真實/虛假預測。這些方法通過堆疊卷積操作以單一尺度從面部區域捕獲偽影。雖然通過堆疊卷積實現了不錯的檢測結果，但它們擅長對局部信息進行建模但失敗了
由於受約束的感受野，全局考慮像素的關係。假設像素之間的關係對於 Deepfake 檢測特別有用，因為某些偽像中的像素明顯不同於圖像中的其餘像素。另一方面，研究者也觀察到偽造圖案的大小各不相同。

For instance, Figure 1 gives examples from popular Deepfake datasets. 

We can see that some forgery traces such as color mismatch occur in small regions (like the mouth corners), while other forgery signals such as visible boundaries that almost span the entire image (see row 3 col 2 in Figure 1). 

Therefore, how to effectively explore regions of different scales in images is extremely critical for Deepfake detection.

To address the above limitations, we explore transformers to model the relationships of pixels due to their strong capability of long-term dependency modeling for both natural language processing tasks and computer vision tasks.

Unlike traditional transformers operating on a single-scale, we propose a multi-scale architecture to capture forged regions that potentially have different sizes. 

Furthermore, suggest that the artifacts of forged images will be destroyed by perturbations such as JPEG compression, making them imperceptible in the RGB domain but can still be detected in the frequency domain. 

This motivates us to use frequency information as a complementary modality in order to reveal artifacts that are no longer perceptible in the RGB domain.

Figure 2: Overview of the proposed M2TR.

The input is a suspicious face image (H x W x C), and the output includes both a forgery detection result and a predicted mask (H x W x 1), which locates the forgery regions.

To this end, we introduce M2TR, a Multi-modal Multi-scale Transformer, for Deepfake detection. M2TR is a multimodal framework, consisting of a Multi-scale Transformer (MT) module and a Cross Modality Fusion (CMF) module. 

In particular, M2TR first extracts features of an input image with a few convolutional layers.

We then generate patches of different sizes from the feature map, which are used as inputs to different heads of the transformer. 

Similarities of spatial patches across different scales are calculated to capture the inconsistency among different regions at multiple scales.

This benefits the discovery of forgery artifacts, since certain subtle forgery clues, e.g., blurring and color inconsistency, are often times hidden in small local patches. 

The outputs from the multi-scale transformer are further augmented with frequency information to derive fused feature representations using a cross modality fusion module. 

Finally, the integrated features are used as inputs to several convolutional layers to generate prediction results. 

In addition to binary classification, we also predict the manipulated regions of the face image in a multi-task manner. 

The rationale behind is that binary classification tends to result in easily overfitted models.

Therefore, we use face masks as additional supervisory signals to mitigate overfitting.

The availability of large-scale training data is an essential factor in the development of Deepfake detection methods.

Existing Deepfake datasets include the UADFV dataset, the DeepFake-TIMIT dataset (DF-TIMIT), the FaceForensics++ dataset (FF++), the Google DeepFake detection dataset (DFD), the FaceBook DeepFake detection challenge (DFDC) dataset, the WildDeepfake dataset, and the Celeb-DF dataset. 

However, the quality of visual samples in current Deepfake datasets is limited, containing clear artifacts (see Figure 1) like color mismatch, shape distortion, visible boundaries, and facial blurring.

Therefore, there is still a huge gap between the images in existing datasets and forged images in the wild which are circulated on the Internet.

Although the visual quality of Celeb-DF is relatively high compared to others, they use only one face swapping method to generate forged images, lacking sample diversity. 

In addition, there are no unbiased and comprehensive evaluation metrics to measure the quality of Deepfake datasets, which is not conducive to the development of subsequent Deepfake research.

In this paper, we present a large-scale and high-quality Deepfake dataset, Swapping and Reenactment DeepFake (SR-DF) dataset, which is generated using the state-of-the-art face swapping and facial reenactment methods for the development and evaluation of Deepfake detection methods. 

We visualize in Figure 4 the sampled forged faces in the proposed SR-DF dataset.

Besides, we propose a set of evaluation criteria to measure the quality of Deepfake
datasets from different perspectives.

We hope the release of SR-DF dataset and the evaluation systems will benefit the future research of Deepfake detection.

Our work makes the following key contributions:

- We propose a Multi-modal Multi-scale Transformer (M2TR) for Deepfake forensics, which uses a multi-scale transformer to detect local inconsistency at different scales and leverages frequency features to improve the robustness of detection. Extensive experiments demonstrate that our method achieves state-of-the-art detection performance on different datasets.

- We introduce a large-scale and challenging Deepfake dataset SR-DF, which is generated with state-of-the-art face swapping and facial reenactment methods.

- We construct the most comprehensive evaluation system and demonstrate that SR-DF dataset is well-suited for the
training Deepfake detection methods due to its visual quality and diversity.

## 2. RELATED WORK

- Deepfake Generation

- Deepfake Detection 

- Visual Transformers

## 3. APPROACH

Figure 3: Illustration of the Multi-scale Transformer.

3.1 Multi-scale Transformer

3.2 Cross Modality Fusion

3.3 Loss functions

- Cross-entropy loss.

- Segmentation loss

- Contrastive loss

## 4. SR-DF DATASET

4.1 Dataset Construction

- Synthesis Approaches

- Post-processing

Figure 4: Example frames from the SR-DF dataset. The first two rows are generated by manipulating facial expressions: (a) First-order-motion and (b) IcFace, while the last two rows are generated by manipulating facial identity: (c) FaceShifter and (d) FSGAN.


Figure 5: Synthesized images of blending the altered face into the background image.We compare three blending methods: naive stitching (left), stitching with color transfer (middle), and stitching with DoveNet (right).

Table 1: A comparison of SR-DF dataset with existing datasets for Deepfake detection. LQ: low-quality, HQ: highquality.

- Existing Deepfake Datasets

4.2 Visual Quality Assessment

- Mask-SSIM

- Perceptual Loss

Table 2: Average Mask-SSIM scores and perceptual loss of different Deepfake datasets.

The value of Mask-SSIM is in the range of [0,1], with the higher value corresponding to better image quality.

We follow to calculate MaskSSIM on videos that we have exact corresponding correspondences for DFD and DFDC dataset. 

For perceptual loss, lower value indicates the better image quality.

Figure 6: A feature perspective comparison of Celeb-DF, FF++ dataset (RAW) and SR-DF dataset. 

We use an ImageNetpretrained ResNet-18 network to extract features and t-SNE for dimension reduction.

Note that we only select one frame in each video for visualization.

Table 3: Average 𝐸𝑤𝑎𝑟𝑝 values of different datasets, with lower value corresponding to smoother temporal results.

We also calculate the 𝐸𝑤𝑎𝑟𝑝 of pristine videos in our dataset.

- Ewarp

- Feature Space Distribution

## 5. EXPERIMENTS

5.1 Experimental Settings

- Datasets

- Evaluation

- Implementation Details

5.2 Evaluation on FaceForensics++

Table 4: Quantitative frame-level detection results on FaceForensics++ dataset under all quality settings. 

The best results are marked as bold.

5.3 Evaluation on Celeb-DF and SR-DF

5.4 Generalization Ability

Table 5: Frame-level AUC scores (%) of various Deepfake detection methods on Celeb-DF and SR-DF dataset.

5.5 From Frames to Videos

5.6 Ablation study

Table 6: AUC scores (%) for cross-dataset evaluation on FF++, Celeb-DF, and SR-DF datasets. 

Note that some methods have not made their code public, so we directly use the data reported in their paper.

“−” denotes the results are unavailable.

- Effectiveness of Different Components

Table 7: Quantitative video-level detection results on different versions of FF++ dataset and SR-DF dataset.

M2TR mean denotes averaging the extracted features obtained by M2TR for all frames as the video-level representation, while M2TR vt f denotes using VTF Block for temporal fusion.

The best results are marked as bold.

Table 8: Ablation results on FF++ (HQ) and FF++ (LQ) with and without Multi-scale Transformer and CMF.

- Effectiveness of the Multi-scale Design

- Effectiveness of the Contrastive Loss

Table 9: Ablation results on FF++ (HQ) using multi-scale Transformer (MT) or single-scale transformer.

Table 10: AUC (%) for cross-dataset evaluation on FF++ (HQ), Celeb-DF, and SR-DF with (denoted as M2TR) and without (denoted as M2TR ncl) the supervision of constrative loss.

## 6. CONCLUSION
