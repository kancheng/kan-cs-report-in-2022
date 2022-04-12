# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Marra F, Gragnaniello D, Cozzolino D, Verdoliva L. Detection of GAN-generated fake images over social networks. In: Proc. of the IEEE Conf. on Multimedia Information Processing and Retrieval (MIPR). IEEE, 2018. 384−389.

Marra, F., Gragnaniello, D., Cozzolino, D., & Verdoliva, L. (2018, April). Detection of gan-generated fake images over social networks. In 2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR) (pp. 384-389). IEEE.

Link : https://ieeexplore.ieee.org/document/8397040

Note : 模拟了篡改图片在社交网络的场景中的检测,结果显示,现有的检测器在现实网络对抗环境下(未知压缩和未知类型等)表现很差

```
The diffusion of fake images and videos on social networks is a fast growing problem.

Commercial media editing tools allow anyone to remove, add, or clone people and objects, to generate fake images.

Many techniques have been proposed to detect such conventional fakes, but new attacks emerge by the day.

Image-to-image translation, based on generative adversarial networks (GANs), appears as one of the most dangerous, as it allows one to modify context and semantics of images in a very realistic way.

In this paper, we study the performance of several image forgery detectors against image-to-image translation, both in ideal conditions, and in the presence of compression, routinely performed upon uploading on social networks.

The study, carried out on a dataset of 36302 images, shows that detection accuracies up to 95% can be achieved by both conventional and deep learning detectors, but only the latter keep providing a high accuracy, up to 89%, on compressed data.
```

社交網絡上虛假圖像和視頻的傳播是一個快速增長的問題，商業媒體編輯工具允許任何人刪除、添加或克隆人和對象，以生成虛假圖像。已經提出了許多技術來檢測這種傳統的偽造品，但新的攻擊每天都在出現。基於生成對抗網絡 (GAN) 的圖像到圖像轉換似乎是最危險的方法之一，因為它允許人們以非常現實的方式修改圖像的上下文和語義。在該研究中，研究者們研究了幾種圖像偽造檢測器對圖像到圖像轉換的性能，無論是在理想條件下，還是在存在壓縮的情況下，通常在上傳到社交網絡時執行。該研究在 36302 張圖像的數據集上進行，表明傳統和深度學習檢測器都可以實現高達 95% 的檢測精度，但只有後者在壓縮數據上保持高達 89% 的高精度。

Bibliography

```
@inproceedings{marra2018detection,
  title={Detection of gan-generated fake images over social networks},
  author={Marra, Francesco and Gragnaniello, Diego and Cozzolino, Davide and Verdoliva, Luisa},
  booktitle={2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)},
  pages={384--389},
  year={2018},
  organization={IEEE}
}
```

Figure 1. Spot the fake. 

Two satellite images, one downloaded from Google Maps, the other artificially generated.

Figure 2. Cycle-GAN based image-to-image translation.

圖 1. 識別假貨。兩張衛星圖像，一張是從谷歌地圖下載的，另一張是人工生成的。

圖 2. 基於 Cycle-GAN 的圖像到圖像轉換。


## I. INTRODUCTION


With the capillary diffusion of social networks, spreading information (and doctored information) has become very easy. 

Fake news are often supported by multimedia contents, aimed at increasing their credibility. 

Indeed, by using powerful image editing tools, such as Photoshop or GIMP, even non-expert users can easily modify an image obtaining realistic results, which evade the scrutiny of human observers. 

A study recently published in Cognitive Research tried to measure people’s ability to recognize, by visual inspection, whether a photo had been doctored or not.

Only 62%-66% of the photos were correctly classified, and users proved even worse at localizing the manipulation.

In a similar study only 58% of the images were correctly classified, and only 46% of the manipulated images were identified as such.

The threat represented by widespread image forgery has stimulated intense research in multimedia forensics.

As a result, automatic algorithms, under suitable hypotheses, achieve a much better detection performance than humans. 

For example, in the first IEEE Image Forensics Challenge, detection accuracies beyond 90% were obtained by means of a machine learning approach with a properly trained classifier.

The situation is almost reversed for what concerns computer generated fake images.

A recent study proved that, for such images, human judgement is significantly better than machine learning.

This may be attributed to the inability of current computer graphics tools to provide a good level of photorealism. 

As a consequence, the research activity in this field has been less intense.

Some papers propose to detect computer graphics images based on statistics extracted from their wavelet decomposition, or from residual images. 

Other papers rely on the different noise introduced by the recording device, on traces of chromatic aberrations, or traces of demosaicing filters. 

Differences in color distribution are explored in, while in the statistical properties of local edge patches are used for discrimination.

In face asymmetry is proposed as a discriminative feature to tell apart computer generated from natural faces.

Only very recently, deep learning has been used for this task and found to outperform preceding approaches.

So, when fake images are easily detected by humans, the need for sophisticated detectors is less impelling.

However, computer graphics technology progresses at a fast pace, and observers find more and more difficult to distinguish between computer-generated and photographic images.

Indeed, new forms of image manipulation based on computer graphics have been recently devised, characterized by a much higher level of photorealism. 

In particular, we focus, here, on image-to-image translation, a process that modifies the attributes of the target image by translating one possible representation of a scene into another one. 

This attack has become relatively easy, and very popular, with the advent of generative adversarial networks (GANs), whose

very aim is to generate images indistinguishable from the real images drawn from a given source. 

Fig.1 provides an example of image-to-image translation.

In this work we analyze the performance of a number of learning-based methods in the detection of image-to-image translation. 

Our goal is to understand if, to which extent, and in which conditions, these attacks can be unveiled.

To this end we will consider several solutions, based both on state-of-the-art methods taken from the image forensic literature, and on general purpose very deep convolutional neural networks (CNNs) properly trained for this task.

We will also study the case in which images are posted over a social network, like Twitter.

In fact, this is both the most common and most challenging situation, since the compression routinely performed upon image uploading tends to impair the performance of forgery detectors.

In the following Sections, we will describe the generation process of the images analyzed in the paper, referring to the method proposed in J. Y. Zhu et al., present the approaches used for detection, and finally discuss the experimental results.


隨著社交網絡的擴散，傳播信息和篡改的傳播信息變得非常容易，其假新聞通常由多媒體內容支持，旨在提高其可信度。事實上，通過使用強大的圖像編輯工具，例如 Photoshop 或 GIMP，即使是非專業用戶也可以輕鬆地修改圖像以獲得逼真的結果，從而避開人類觀察者的審查。最近發表在 Cognitive Research 上的一項研究試圖通過視覺檢查來衡量人們識別照片是否被篡改的能力。只有 62%-66% 的照片被正確分類，而且用戶在定位操作方面表現得更差。在一項類似的研究中，只有 58% 的圖像被正確分類，並且只有 46% 的經過處理的圖像被正確分類，而廣泛的圖像偽造所代表的威脅刺激了多媒體取證的深入研究。結果，在適當的假設下，自動算法可以實現比人類更好的檢測性能，例如在第一屆 IEEE 圖像取證挑戰賽中，通過機器學習方法和經過適當訓練的分類器獲得了超過 90% 的檢測精度。

同時對於計算機生成的虛假圖像，情況幾乎相反，最近的一項研究證明，對於此類圖像，人類的判斷明顯優於機器學習，這可能歸因於當前的計算機圖形工具無法提供良好的照片級真實感。因此該領域的研究活動較少，一些論文提出基於從小波分解或殘差圖像中提取的統計數據來檢測計算機圖形圖像，而其他論文依賴於記錄設備引入的不同噪聲、色差痕跡或去馬賽克濾波器的痕跡。探討了顏色分佈的差異，而在局部邊緣塊的統計特性中則用於區分，人臉不對稱被提出作為區分計算機從自然人臉生成的判別特徵。直到最近深度學習才被用於這項任務，並被發現優於以前的方法。因此，當虛假圖像很容易被人類檢測到時，對複雜檢測器的需求就不那麼迫切了。然而，計算機圖形技術發展迅速，觀察者發現越來越難以區分計算機生成的圖像和攝影圖像。事實上，最近已經設計出基於計算機圖形的新形式的圖像處理，其特點是更高水平的照片寫實。

特別是，研究者在這里關注圖像到圖像的轉換，這是一個通過將場景的一種可能表示轉換為另一種表示來修改目標圖像屬性的過程，隨著生成對抗網絡 (GAN) 的出現，這種攻擊變得相對容易且非常流行。其目的是生成與從給定來源繪製的真實圖像無法區分的圖像，研究者圖中提供了圖像到圖像轉換的示例。在這項工作中，研究者分析了許多基於學習的方法在檢測圖像到圖像轉換中的性能，其目標是了解這些攻擊是否、在何種程度上以及在何種條件下可以被揭露。

為此研究者將考慮幾種解決方案，這些解決方案既基於從圖像取證文獻中獲取的最先進的方法，也基於針對該任務進行適當訓練的通用非常深的捲積神經網絡 (CNN)。同時研究者還將研究通過社交網絡（如 Twitter）發布圖像的情況。事實上，這既是最常見也是最具挑戰性的情況，因為在圖像上傳時常規執行的壓縮往往會削弱偽造檢測器的性能。最後研究者將參考 J. Y. Zhu 等人提出的方法，描述本文分析的圖像的生成過程，介紹用於檢測的方法，最後討論實驗結果。

## V. CONCLUSIONS

We have presented a study on the detection of images manipulated by GAN-based image-to-image translation. 

Several detectors perform very well on original images, but some of them show dramatic impairments on Twitter-like compressed images. 

Robustness is better preserved by deep networks, especially XceptionNet, which keeps working reasonably well even in the presence of training-test mismatching.

Future research, besides extending the analysis to more manipulations and detectors, will study crossmethod performance, possibly after transfer learning, w.r.t. other synthetic image generators. Moreover, we will test the performance in real world scenarios involving different social networks.

該研究提出了一項關於檢測由基於 GAN 的圖像到圖像轉換操作的圖像的研究，比如一些檢測器在原始圖像上的表現非常好，但其中一些在類似 Twitter 的壓縮圖像上表現出明顯的缺陷。深度網絡可以更好地保持魯棒性，尤其是 XceptionNet，即使在訓練-測試不匹配的情況下，它也能保持相當好的工作。
