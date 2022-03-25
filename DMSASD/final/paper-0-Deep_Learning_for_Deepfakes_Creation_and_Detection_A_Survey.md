# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Deep Learning for Deepfakes Creation and Detection: A Survey

https://www.researchgate.net/publication/336055871_Deep_Learning_for_Deepfakes_Creation_and_Detection_A_Survey

有持續更新，但是該版本是早期的版本。

- list1 - 105 ,2022 v4, https://arxiv.org/pdf/1909.11573.pdf

```
Deep learning has been successfully applied to solvevarious complex problems ranging from big data analytics tocomputer vision and human-level control. 

Deep learning advanceshowever have also been employed to create software that cancause threats to privacy, democracy and national security. 

Oneof those deep learning-powered applications recently emergedis “deepfake”. 

Deepfake algorithms can create fake images andvideos that humans cannot distinguish them from authenticones.

The proposal of technologies that can automatically detectand assess the integrity of digital visual media is thereforeindispensable.

This paper presents a survey of algorithms usedto create deepfakes and, more importantly, methods proposed todetect deepfakes in the literature to date.

We present extensivediscussions on challenges, research trends and directions relatedto deepfake technologies.

By reviewing the background of deep-fakes and state-of-the-art deepfake detection methods, this studyprovides a comprehensive overview of deepfake techniques andfacilitates the development of new and more robust methods todeal with the increasingly challenging deepfakes.
```

深度學習已成功應用於解決從大數據分析到計算機視覺和人類水平控制的各種複雜問題，其技術也可能會被用於創造出可能對隱私、民主體制和國家安全造成威脅的軟體應用。 而近來出現的基於深度學習的應用程序則是 “deepfake”，其演算法可以做出人類用肉眼無法將它們與真品區分開來的假影像和影片。因此，擁有能夠自動檢測和評估數位視覺媒體完整性等技術的提議是必不可少的。本研究介紹了用於創造深度偽造的演算法的調查，更重要的是，介紹了迄今為止文獻中提出的用於檢測深度偽造的方法。同時對與深度偽造技術相關的挑戰、研究趨勢和方向進行了廣泛的討論。通過回顧深造假的背景和最先進的深造假檢測方法，本研究提供了深造假技術的全面概述，並有助於開發新的、更強大的方法來應對日益具有挑戰性的深度偽造局面。

Bibliography

```
@unknown{unknown,
author = {Nguyen, Thanh and Nguyen, Cuong M. and Nguyen, Tien and Nguyen, Duc and Nahavandi, Saeid},
year = {2019},
month = {09},
pages = {},
title = {Deep Learning for Deepfakes Creation and Detection: A Survey}
}
```

Note : 2019 年只有针对早期图像篡改工作的一些总结，該 arXiv 有持續更新至 2022。

```
Deep learning has been successfully applied to solve various complex problems ranging from big data analytics to computer vision and human-level control. 

Deep learning advances however have also been employed to create software that can cause threats to privacy, democracy and national security. 

One of those deep learning-powered applications recently emerged is deepfake. 

Deepfake algorithms can create fake images and videos that humans cannot distinguish them from authentic ones. 

The proposal of technologies that can automatically detect and assess the integrity of digital visual media is therefore indispensable. 

This paper presents a survey of algorithms used to create deepfakes and, more importantly, methods proposed to detect deepfakes in the literature to date. 

We present extensive discussions on challenges, research trends and directions related to deepfake technologies. 

By reviewing the background of deepfakes and state-of-the-art deepfake detection methods, this study provides a comprehensive overview of deepfake techniques and facilitates the development of new and more robust methods to deal with the increasingly challenging deepfakes.
```

深度學習已成功應用於解決從大數據分析到計算機視覺和人類水平控制的各種複雜問題，然而，深度學習的進步也被用於創建可能對隱私、民主和國家安全造成威脅的軟件，最近出現的深度學習驅動的應用之一是 deepfake。其 Deepfake 演算法可以創建人類無法區分真實圖像和視頻的虛假圖像和視頻，因此，能夠自動檢測和評估數字視覺媒體完整性的技術的提議是必不可少的，該研究介紹了用於創建深度偽造的算法的調查，更重要的是，介紹了迄今為止文獻中提出的用於檢測深度偽造的方法。研究者對與深度偽造技術相關的挑戰、研究趨勢和方向進行了廣泛的討論。通過回顧 deepfake 的背景和最先進的 deepfake 檢測方法，本研究提供了 deepfake 技術的全面概述，並有助於開發新的、更強大的方法來處理日益具有挑戰性的 deepfake。

Bibliography

```
@article{nguyen2019deep,
  title={Deep learning for deepfakes creation and detection: A survey},
  author={Nguyen, Thanh Thi and Nguyen, Quoc Viet Hung and Nguyen, Cuong M and Nguyen, Dung and Nguyen, Duc Thanh and Nahavandi, Saeid},
  journal={arXiv preprint arXiv:1909.11573},
  year={2019}
}
```

1. Introduction

In a narrow definition, deepfakes (stemming from “deep learning” and “fake”) are created by techniques that can superimpose face images of a target person onto a video of a source person to make a video of the target person doing or saying things the source person does.

This constitutes a category of deepfakes, namely faceswap. In a broader definition, deepfakes are artificial intelligence-synthesized content that can also fall into two other categories, i.e., lip-sync and puppet-master. 

Lip-sync deepfakes refer to videos that are modified to make the mouth movements consistent with an audio recording. 

Puppet-master deepfakes include videos of a target person (puppet) who is animated following the facial expressions, eye and head movements of another person (master) sitting in front of a camera [1].

While some deepfakes can be created by traditional visual effects or computer-graphics approaches, the recent common underlying mechanism for deepfake creation is deep learning models such as autoencoders and generative adversarial networks (GANs), which have been applied widely in the computer vision domain [2–8]. 

These models are used to examine facial expressions and movements of a person and synthesize facial images of another person making analogous expressions and movements [9].

Deepfake methods normally require a large amount of image and video data to train models to create photo-realistic images and videos. 

As public figures such as celebrities and politicians may have a large number of videos and images available online,
they are initial targets of deepfakes. 

Deepfakes were used to swap faces of celebrities or politicians to bodies in porn images and videos.

The first deepfake video emerged in 2017 where face of a celebrity was swapped to the face of a porn actor. 

It is threatening to world security when deepfake methods can be employed to create videos of world leaders with fake speeches for falsification purposes [10–12]. 

Deepfakes therefore can be abused to cause political or religion tensions between countries, to fool public and affect results in election campaigns, or create chaos in financial markets by creating fake news [13–15]. 

It can be even used to generate fake satellite images of the Earth to contain objects that do not really exist to confuse military analysts, e.g., creating a fake bridge across a river although there is no such a bridge in reality. 

This can mislead a troop who have been guided to cross the bridge in a battle [16, 17].

As the democratization of creating realistic digital humans has positive implications, there is also positive use of deepfakes such as their applications in visual effects, digital avatars, snapchat filters, creating voices of those who have lost theirs or updating episodes of movies without reshooting them [18]. 

Deepfakes can have creative or productive impacts in photography, video games, virtual reality, movie productions, and entertainment, e.g., realistic video dubbing of foreign films, education through the reanimation of historical
figures, virtually trying on clothes while shopping, and so on [19, 20].

However, the number of malicious uses of deepfakes largely dominates that of the positive ones.

The development of advanced deep neural networks and the availability of large amount of data have made the forged images and videos almost indistinguishable to humans and even to sophisticated computer algorithms.

The process of creating those manipulated images and videos is also much simpler today as it needs as little as an identity photo or a short video of a target individual. 

Less and less effort is required to produce a stunningly convincing tempered footage. 

Recent advances can even create a deepfake with just a still image [21].

Deepfakes therefore can be a threat affecting not only public figures but also ordinary people. For example, a voice deepfake was used to scam a CEO out of $243,000 [22]. 

A recent release of a software called DeepNude shows more disturbing threats as it can transform a person to a non-consensual porn [23]. 

Likewise, the Chinese app Zao has gone viral lately as less-skilled users can swap their faces onto bodies of movie stars and insert themselves into well-known movies and TV clips [24]. 

These forms of falsification create a huge threat to violation of privacy and identity, and affect many aspects of human lives.

Finding the truth in digital domain therefore has become increasingly critical. It is even more challenging when dealing with deepfakes as they are majorly used to serve malicious purposes and almost anyone can create deepfakes these days using existing deepfake tools.


Thus far, there have been numerous methods proposed to detect deepfakes [25–29]. 

Most of them are based on deep learning, and thus a battle between malicious and positive uses of deep learning methods has been arising. 

To address the threat of face-swapping technology or deepfakes, the United States Defense Advanced Research Projects Agency (DARPA) initiated a research scheme in media forensics (named Media Forensics or MediFor) to accelerate the development of fake digital visual media detection methods [30]. 

Recently, Facebook Inc. teaming up with Microsoft Corp and the Partnership on AI coalition have launched the Deepfake Detection Challenge to catalyse more research and development in detecting and preventing deepfakes from being used to mislead viewers [31]. 

Data obtained from https://app.dimensions.ai at the end of 2021 show that the number of deepfake papers has increased significantly in recent years (Fig. 1). 

Although the obtained numbers of deepfake papers may be lower than actual numbers but the research trend of this topic is obviously increasing.


Fig. 1. Number of papers related to deepfakes in years from 2016 to 2021, obtained from https://app.dimensions.ai at the end of 2021 with the search keyword “deepfake” applied to full text of scholarly papers.

This paper presents a survey of methods for creating as well as detecting deepfakes. 

There have been existing survey papers about this topic in [19, 20, 32], we however carry out the survey with different perspective and taxonomy. 

For example, Mirsky and Lee [19] focused on reenactment approaches (i.e., to change a target’s expression, mouth, pose, gaze or body), and replacement approaches (i.e., to replace a target’s face by swap or transfer methods).

Verdoliva [20] separated detection approaches into conventional methods (e.g., blind methods without using any external data for training, one-class sensor-based and model-based methods, and supervised methods with handcrafted features)
and deep learning-based approaches (e.g., CNN models). 


Tolosana et al. [32] categorized both creation and detection methods based on the way deepfakes are created, including entire face synthesis, identity swap, attribute manipulation, and expression swap.

We, on the other hand, focus on the technical perspective of deepfakes. 

We first present the principles of deepfake algorithms and how deepfakes are created technically by deep learning approaches such as autoencoder and GAN models in Section 2. 

In Section 3, we categorize the deepfake detection methods based on the data type they take into account, i.e., images or videos. 

With fake video detection methods, two main categories are identified based on whether the method uses temporal features across frames or visual artifacts within a video frame. 

We then discuss in detail challenges, research trends and directions on deepfake detection and multimedia forensics problems in Section 4.

2. Deepfake Creation

Deepfakes have become popular due to the quality of tampered videos and also the easy-to-use ability of their applications to a wide range of users with various computer skills from professional to novice. 

These applications are mostly developed based on deep learning techniques.

Deep learning is well known for its capability of representing complex and high-dimensional data.

One variant of the deep networks with that capability is deep autoencoders, which have been widely
applied for dimensionality reduction and image compression [33–35]. 

The first attempt of deepfake creation was FakeApp, developed by a Reddit user using autoencoder-decoder pairing structure [36, 37].

In that method, the autoencoder extracts latent features of face images and the decoder is used to reconstruct the face images. 

To swap faces between source images and target images, there is a need of two encoder-decoder pairs where each pair is used to train on an image set, and the encoder’s parameters are shared between two network pairs.

In other words, two pairs have the same encoder network.

This strategy enables the common encoder to find and learn the similarity between two sets of face images, which are relatively unchallenging because faces normally have similar features such as eyes, nose, mouth positions. 

Fig. 2 shows a deepfake creation process where the feature set of face A is connected with the decoder B to reconstruct face B from the original face A. 

This approach is applied in several works such as DeepFaceLab [38], DFaker [39], DeepFake tf (tensorflow-based deepfakes) [40].

By adding adversarial loss and perceptual loss implemented in VGGFace [57] to the encoder-decoder architecture, an improved version of deepfakes based on the generative adversarial network [4], i.e., faceswap-GAN, was proposed in [58]. 

The VGGFace perceptual loss is added to make eye movements to be more realistic and consistent with input faces and help to smooth out artifacts in segmentation mask, leading to higher quality output videos. 

This model facilitates the creation of outputs with 64x64, 128x128, and 256x256 resolutions.
In addition, the multi-task convolutional neural network (CNN) from the FaceNet implementation [59] is used to make face detection more stable and face alignment more reliable.

The CycleGAN [60] is utilized for generative network implementation in this model.

Fig. 2. A deepfake creation model using two encoder-decoder pairs.

Two networks use the same encoder but different decoders for training process (top). An image of face A is encoded with the common encoder and decoded with decoder B to create a deepfake (bottom).

The reconstructed image (in the bottom) is the face B with the mouth shape of face A.

Face B originally has the mouth of an upside-down heart while the reconstructed face B has the mouth of a conventional
heart Fig. 3.

The GAN architecture consisting of a generator and a discriminator, and each can be implemented by a neural network. The entire system can be trained with backpropagation that allows both networks to improve their capabilities.

A conventional GAN model comprises two neural networks: a generator and a discriminator as depicted in Fig. 3. 

Given a dataset of real images x having a distribution of pdata, the aim of the generator G is to produce images G(z) similar to real images x with z being noise signals having a distribution of pz. 

The aim of the discriminator G is to correctly classify images generated by G and real images x. The discriminator D is trained to improve its classification capability, i.e., to maximize D(x), which represents the probability that x is a real image rather than a fake image generated by G.

On the other hand, G is trained to minimize the probability that its outputs are classified by D as synthetic images, i.e., to minimize 1 − D(G(z)). 

This is a minimax game between two players D and G that can be described by the following value function [4]:

$$
\begin{aligned}
\min _{G} \max _{D} V(D, G)=& \mathbb{E}_{x \sim p_{\text {data }}(x)}[\log D(x)] \\
&+\mathbb{E}_{z \sim p_{z}(z)}[\log (1-D(G(z)))]
\end{aligned}
$$

After sufficient training, both networks improve their capabilities, i.e., the generator G is able to produce images that are really similar to real images while the discriminator D is highly capable of distinguishing fake images from real ones.

Fig. 4. A structure comparison between two generators: one of a
PGGAN [61] (a) and another of a StyleGAN [51] (b). In PGGAN,
the latent code is fed to the input layer only. In StyleGAN, the latent code is first mapped into an intermediate latent space W, which
is then injected into the generator via the adaptive instance normalization (AdaIN) at each convolution layer. Gaussian noise is added after
each convolution, but before the AdaIN operations [51].

Table 1 presents a summary of popular deepfake tools
and their typical features. Among them, a prominent
method for face synthesis based on a GAN model,
namely StyleGAN, was introduced in [51]. 

StyleGAN is motivated by style transfer literature [62] with a special generator network architecture that is able to create realistic face images.



