# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Zeinali H, Stafylakis T, Athanasopoulou G, Rohdin J, Gkinis I, Burget L, Cernocky JH. Detecting spoofing attacks using VGG and SincNet: BUT-Omilia submission to ASVspoof 2019 challenge. In: Proc. of the 20th Annual Conf. of the Int’l Speech Communication Association. 2019. 1073−1077.

Zeinali, H., Stafylakis, T., Athanasopoulou, G., Rohdin, J., Gkinis, I., Burget, L., & Černocký, J. (2019). Detecting spoofing attacks using vgg and sincnet: but-omilia submission to asvspoof 2019 challenge. arXiv preprint arXiv:1907.12908.

Link : https://arxiv.org/abs/1907.12908

Note : 應用不同架構來應對攻擊

Tag : CVPR

```
In this paper, we present the system description of the joint efforts of Brno University of Technology (BUT) and Omilia - Conversational Intelligence for the ASVSpoof2019 Spoofing and Countermeasures Challenge.

The primary submission for Physical access (PA) is a fusion of two VGG networks, trained on single and two-channels features.

For Logical access (LA), our primary system is a fusion of VGG and the recently introduced SincNet architecture.

The results on PA show that the proposed networks yield very competitive performance in all conditions and achieved 86\:\% relative improvement compared to the official baseline.

On the other hand, the results on LA showed that although the proposed architecture and training strategy performs very well on certain spoofing attacks, it fails to generalize to certain attacks that are unseen during training.

```

該研究介紹了布爾諾理工大學 (BUT) 和 Omilia 共同努力的系統描述 — ASVSpoof2019 Spoofing and Countermeasures Challenge 的對話智能，其物理訪問（PA）的主要提交是兩個 VGG 網絡的融合，並在單通道和雙通道特徵上進行了訓練。對於邏輯訪問 (LA)，研究者的主要系統是 VGG 和最近引入的 SincNet 架構的融合，其 PA 上的結果表明，所提出的網絡在所有條件下都產生了非常有競爭力的性能，並且與官方基線相比實現了 86\:\% 的相對改進。另一方面，LA 上的結果表明，儘管所提出的架構和訓練策略在某些欺騙攻擊上表現得非常好，但它無法推廣到在訓練期間看不見的某些攻擊下。

Bibliography

```
@article{zeinali2019detecting,
  title={Detecting spoofing attacks using vgg and sincnet: but-omilia submission to asvspoof 2019 challenge},
  author={Zeinali, Hossein and Stafylakis, Themos and Athanasopoulou, Georgia and Rohdin, Johan and Gkinis, Ioannis and Burget, Luk{\'a}{\v{s}} and {\v{C}}ernock{\`y}, Jan and others},
  journal={arXiv preprint arXiv:1907.12908},
  year={2019}
}
```

## 1. Introduction

To facilitate better and safer customer support in e.g. banking and call centers, there is a growing demand for convenient and robust automatic authentication systems. 

Automatic speaker verification (ASV) a.k.a. voice biometrics is arguably the most natural and least intrusive authentication method in such applications. 

Unfortunately, ASV systems are vulnerable to synthetic speech, created by text-to-speech (TTS) and voice conversion (VC) methods, and to replay/presentation attacks [1]. 

The attempts to deceive an ASV system by such methods are known as ASV spoofing attacks. 

While research in ASV has been ongoing for several decades, it is only in the recent years that the research community has started to tackle spoofing attacks systematically, through a series of ASV spoofing and countermeasures challenges [2, 3].


Spoofing attacks to ASV systems can be categorized into 4 types [1].

The first one is impersonation which can be rejected by an accurate ASV system [4]. 

The second and third types are TTS and VC which were tackled in the ASVspoof 2015 challenge [2] and several methods have been proposed to detect them [5, 6, 7, 8]. 

The last type of attacks is replay attack with pre-recorded audio and it is considered to be the most difficult attack to detect [1].

Possible ways to tackle this problem are (a) anti-spoofing techniques based on detecting typical distortions in recorded and replayed audio [3, 9], (b) using audio fingerprinting [10] to detect a replay of an enrollment utterance, and (c) using liveness detection and phrase verification [11] in text-dependent speaker verification.

This paper presents the collaborative efforts of BUT and Omilia to introduce novel countermeasures for the last three attack types, as part of the 2019 automatic speaker verification (ASV) anti-spoofing challenge. 

All our systems are based on deep neural network (DNN) architectures, trained to discriminate between bonafide and synthetic or replayed speech and are employed as end-to-end classifiers, i.e. without any external backend. 

The physical access (PA) system is a fusion of two VGG [12] networks using different features, while the logical access (LA) system is a fusion of one VGG network and two SincNet networks [13].

## 2. Physical access

2.1. Features and preprocessing

For this challenge we explore several features such as Mel-filter bank, MFCC, constant Q-transform (CQT) [14], CQCC [15], and power spectrogram.

Among the explored features, power spectrogram yields superior performance, followed by CQT features. 

Accordingly, we use these two features in most of our experiments. In particular, the submitted systems use either the power spectrograms as a single input channel, or both the power spectrograms and the CQT features fed as two different input channels. 

As a feature preprocessing, both CQT and power spectrogram are first transferred to log domain and then subjected to mean and variance normalization (MVN) before being fed to the network.

2.2. Example and minibatch generation for network training

The procedure for generating training examples and minibatches can greatly affect the performance of neural networks in audio processing. 

Therefore, we experimented with several different strategies for this. 

For example generation, we first concatenate all features of the same class (same attack id) and speaker. 

We then split the concatenated features into small segments of the same size.

Initially we used four second segments but after doing several experiments, we found that networks trained on smaller segments performed better than those trained on large segments, mainly because they overfit less to the training data. 

The size of the examples used to train the submitted systems is one second (i.e. 100 frames).

For minibatch generation we experimented with different strategies for distributing the examples into minibatches.

We found that the best strategy is to only use examples from a single speaker within each minibatch (a few minibatches may contain examples from more speakers in order to use all training
data). 

Each minibatch has 128 examples. 

After each epoch, we randomise the examples and generate the minibatches again for better generalization.

2.3. Training and development data

For training the networks, the official training set of the challenge was used. 

This set contains audio samples from 20 speakers. One of the speakers was randomly selected for network training validation set which is roughly 5 % of the training data.


The development set is also the official challenge’s development set. This set which contains 20 speakers, was only used for evaluating networks and comparing different methods and
training strategies.

2.4. Networks and training strategies

For this challenge, two different topologies were used for Physical access.

The first one is a modified version of a VGG network [12] which has shown good performance in Audio Tagging and Audio Scene Classification [16, 17]. 

The second network is a modified version of a Light CNN (LCCN) [18] which had the best performance for ASVSpoof2017 challenge [9].

We have used a modified version of both networks for acoustic scene classification challenge 2019 [19]. 

In the following two sections, both networks will be explained in more detail.

2.4.1. VGG-like network

The VGG network comprises several convolutional and pooling layers followed by a statistics pooling and several dense layers which perform classification. 

Table 1 provides a detailed description of the proposed VGG architecture. 

There are 6 convolutional blocks in the model, each containing 2 convolutional layers and one max-pooling. 

Each max-pooling layer reduce the size of frequency axis to half while only one of them reduces the temporal resolution. 

After the convolutional layers, there is a mean pooling layer which operates only on the time axis and calculates the mean over time. 

After this layer, there is a flatten layer which simply concatenates the 4 remaining frequency channels. 

Finally there are 3 dense layers which perform the classification task.

2.4.2. Light CNN (LCNN)

Table 2 shows the used LCNN topology for this challenge. 

This network is a combination of convolutional and max-pooling layers and uses Max-Feature-Map (MFM) as non-linearity.

MFM is a layer which simply reduce the number of output channels to the half by taking the maximum of two consecutive channels (or any other combination of two channels). 

The rest of this network (statistics and classification parts) is identical to the proposed VGG network.

2.5. Fusion and submitted systems

Since the evaluation protocol does not allow us to estimate fusion parameters on the development set, we choose to use a simple average with equal weight for our best systems. 

Our submissions are the following:

- Primary: Fusion of two VGG networks. The first one is trained using two-channels features while the second one is fed with single channel log-power spectrogram.

- Single best: Our single best system for this part is the VGG network with two-channels features.

- Contrastive 1: This system is a VGG network with single channel log-power spectrogram features.

- Contrastive 2: The second contrastive system is LCNN network again with single channel log-power spectrogram as features.


Table 1: The proposed VGG architecture. Conv2D: two dimensional convolutional layer. 

MeanPooling: a layer which calculate the mean in time axis and reduce the shape (remove the time axis). Dense: fully connected dense layer

## 3. Logical access

3.1. Logical access using SincNet

SincNet is a novel end-to-end neural network architecture, which receives raw waveforms as input rather than handcrafted features such as spectrograms or CQCCs [13]. 

Contrary to other end-to-end approaches, SincNet constrains the first 1D convolutional layer to parametrized Sinc functions, encouraging it to discover more meaningful (band-pass) filters. 

This architecture offers a very efficient way to derive a customized filter bank that is specifically tuned for the desired application.

The filters are initialized using the Mel-frequency filter bank and their low and high cutoff frequencies are adapted with standard backpropagation as any other layer.

SincNet is originally designed for speech and speaker recognition tasks, and we believe it is a good fit for the problem at hand, since certain artifacts created by TTS and VC systems should be more easily detectable in the waveform domain.

3.1.1. SincNet architecture

The first block consists of three convolutional layers. 

The first layer performs Sinc-based convolutions, using 80 filters of length L=251 samples. 

The remaining two layers using 60 filters of length 5. 

Next, three fully-connected layers composed of 2048 neurons and normalized with batch normalization were applied. 

All hidden layers use leaky-ReLU nonlinearities. 

Frame-level binary classification is performed by applying a softmax classifier and cross-entropy criterion. 

We use high dropout rates in all layers in one of our networks, in order to improve its generalizability to unseen speakers and spoofing attacks [13]. 

Our implementation is based on the opensource PyTorch code provided by the authors 1
.
3.1.2. Training and evaluating SincNet

SincNet is trained by randomly sampling 200 ms chunks from each utterance, which are fed into the SincNet architecture.

Mean and variance normalization and energy-based voice activity detector are applied in an utterance-level fashion. 

As in the original SincNet we use RMSprop as optimizer, while we train it with only 5 epochs, each comprising 1000 minibatches of size 256. 

In the first epoch, we use a small learning rate, which we increase and decrease again for the last epoch (namely $10^{−5}$, $10^{−4}$, $10^{−3}$ and $10^{−4}$). 

The small learning rate in the first epoch is chosen in order to preserve the melfrequency based initialization of the Sinc functions. 

This learning rate approach results to a steep decrease in the loss from the fourth epoch.

Moreover, during training we ensure that each minibatch used for back-propagation is balanced, such that for every bonafide sample we randomly select a spoof sample from the same speaker, resulting in 128 bonafide samples and 128 spoof samples for every minibatch.

During evaluation, utterance-level LLRs are derived by averaging the corresponding frame-level LLRs, as estimated by the logarithmic softmax layer.

3.1.3. Cross-validation over presentation attacks

In order to assess the generalizability of the network to novel attacks, we first trained the network on a subset of attacks and evaluated it on the remaining ones. 

By using this crossvalidation scheme, the EER attained on unseen attacks was always below 0.2% EER, underlying the good generalization capacity of the network, at least between those attacks included in the training and development sets.

Finally, we trained the model on the whole training set using the best training strategy defined by the cross-validation and we obtained 0.0 % EER (i.e. no errors) on the full development set.

3.2. Logical access using VGG

For the Logical access we explored the two VGG architectures that were the best for Physical access, i.e. the architecture described in Table 1 with either log-power spectrum as a single input channel, or with log-power-spectrum and CQT as two input channels. 

Using only the log-power spectrum was substantially better than using both features.

It is worth noting that we experimented with the SincNet architecture on presentation attacks (i.e. PA), however its performance was inferior to that of VGG.

3.3. Fusion and submitted systems

As in physical access we have 4 systems and again we fuse them using simple averaging.

- Primary: Our primary system is fusion of a VGG network with single channel log-power spectrogram features and 2 SincNets which differ in the dropout rate.

- Single best: SincNet with the standard dropout rates.

- Contrastive 1: Fusion of two VGG network which were trained using two channel and single channel features like Physical access.

- Contrastive 2: SincNet with high dropout rates.

Table 2: The proposed LCNN architecture. MFM: MaxFeature-Map.


## 4. Experimental results

In this section, we report the official results evaluated by the challenge organizers, based on the scores we submitted.

4.1. Results on Physical Access

Table 5 reports results attained by different submissions for physical access. 

The first row of the table provides the results for the organizers’ baseline which is a GMM based method with CQCC features.

The results on the evaluation set attained by our submitted systems demonstrate their capacity in generalizing very well to new PA configurations. 

By comparing the single best and contrastive1 systems it is evident that the single channel features perform considerably better on the evaluation set (has better generalization).

A more analytic report can be found in Table 3. 

The first letter in attack ID shows the environment definition. 

From A to C, room size, room reverberation time and talker-to-ASV distance are increased and so, detection of A is more difficult than C. 

The second letter of attack ID shows attack definition. 

From A to C, attacker-to-talker distance is increased while replay device quality is decreased. 

Again, A is more difficult than C. It is clear that the trends of the results are in line with expectations in most cases (i.e. AA has the worst results and CC has the best.)

4.2. Results on Logical Access

We present here the results we attained on the evaluation test. 

In Table 6 we report the results on the two sets. 

Clearly, although our systems performed exceptionally well on the development set, failed to generalize well to certain logical attacks unseen in training.

The LA detailed results are reported in Table 4 based on different waveform generation methods include: neural waveform (A01, A08, A10, A12, A15), vocoder (A02, A03, A07, A09, A14, A18), waveform filtering (A05, A13, A17), spectral filtering (A06, A19) and waveform concatenation (A04, A13, A16).

From the table, we observe that the attacks which degraded the performance the most were A10, A12, and A15, which were all based on neural waveform TTS systems. 

It is interesting to note that for these attacks, the EER attained by SincNet was above 50 % (not reported here) while it performs better than or same as the overall best single system in 12 conditions. 

The conclusion is that the cross-validation method we performed was insufficient to prevent the network from overfitting and some more analysis will be needed to figure out why the SincNet totally failed for some waveform generation methods.

## 5. Conclusions

In this paper we presented the joint submission of BUT and Omilia for the ASVspoof 2019. 

For PA, we followed the VGG architecture and obtained very competitive results in both development and evaluation sets, by fusing only two networks. 

For LA, we fused a VGG architecture with the recently proposed SincNet. 

The rationale for employing the latter was its ability to jointly optimize the networks and the feature extractor, which was shown to be very effective for speech and speaker recognition. 

Despite our efforts to prevent overfitting (mainly via attack-level cross validation in training and development), the results on LA showed the difficulty of the SincNet in generalizing to certain attacks which were significantly different to those in the training. 

We conclude that more research is required in order to make full use of end-to-end anti-spoofing architectures such as SincNet in cases of large mismatch between training and evaluation attacks.


Table 3: Physical access detailed results based on min-tDCF for different conditions. 

The first section shows the baseline results and the second section shows the primary and single best results of the best-performing systems, both from team T28.

Table 4: Logical access detailed results based on min-tDCF for different conditions. 

The first section shows the baseline results and the second section shows the primary system results of the best performing team (T05) as well as the overall best single system results
(team T45). 

The bold numbers show conditions where our single system performs better or the same as the best single system.

Table 5: Physical access results of different submissions

Table 6: Logical access results of different submissions


