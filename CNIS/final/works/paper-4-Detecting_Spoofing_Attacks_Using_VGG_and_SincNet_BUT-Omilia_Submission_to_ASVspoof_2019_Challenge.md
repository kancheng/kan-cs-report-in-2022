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

While research in ASV has been ongoing for several decades, it is only in the recent years that the research community has started to tackle spoofing attacks systematically, through a series of ASV spoofing and countermeasures challenges.


Spoofing attacks to ASV systems can be categorized into 4 types.

The first one is impersonation which can be rejected by an accurate ASV system. 

The second and third types are TTS and VC which were tackled in the ASVspoof 2015 challenge and several methods have been proposed to detect them. 

The last type of attacks is replay attack with pre-recorded audio and it is considered to be the most difficult attack to detect.

Possible ways to tackle this problem are (a) anti-spoofing techniques based on detecting typical distortions in recorded and replayed audio, (b) using audio fingerprinting to detect a replay of an enrollment utterance, and (c) using liveness detection and phrase verification in text-dependent speaker verification.

This paper presents the collaborative efforts of BUT and Omilia to introduce novel countermeasures for the last three attack types, as part of the 2019 automatic speaker verification (ASV) anti-spoofing challenge. 

All our systems are based on deep neural network (DNN) architectures, trained to discriminate between bonafide and synthetic or replayed speech and are employed as end-to-end classifiers, i.e. without any external backend. 

The physical access (PA) system is a fusion of two VGG networks using different features, while the logical access (LA) system is a fusion of one VGG network and two SincNet networks.

為了促進更好和更安全的客戶支持，例如銀行和呼叫中心，對方便和強大的自動身份驗證系統的需求不斷增長，自動說話人驗證 (ASV) 又名語音生物識別技術可以說是此類應用中最自然且侵入性最小的身份驗證方法。不幸的是此 ASV 系統容易受到由文本到語音 (TTS) 和語音轉換 (VC) 方法創建的合成語音以及重放/演示攻擊的攻擊。通過這種方法欺騙 ASV 系統的嘗試被稱為 ASV 欺騙攻擊。雖然 ASV 的研究已經進行了幾十年，但直到近幾年，研究界才開始通過一系列 ASV 欺騙和對策挑戰系統地解決欺騙攻擊。

針對 ASV 系統的欺騙攻擊可分為 4 種類型。第一個是模擬，可以被準確的 ASV 系統拒絕，第二種和第三種類型是 TTS 和 VC，它們在 ASVspoof 2015 挑戰中得到解決，並且已經提出了幾種檢測它們的方法。最後一種攻擊是使用預先錄製的音頻進行重放攻擊，它被認為是最難檢測的攻擊。解決這個問題的可能方法是(a)基於檢測錄製和重放音頻中的典型失真的反欺騙技術，(b) 使用音頻指紋檢測註冊話語的重放，以及 (c)使用活躍度檢測和短語驗證在依賴於文本的說話者驗證中。

作為 2019 年自動說話人驗證 (ASV) 反欺騙挑戰的一部分，該研究介紹了 BUT 和 Omilia 為最後三種攻擊類型引入新對策的合作努力。所有的系統都基於深度神經網絡 (DNN) 架構，經過訓練可以區分真實語音和合成或重放語音，並用作端到端分類器，即沒有任何外部後端。物理訪問 (PA) 系統是兩個使用不同特徵的 VGG 網絡的融合，而邏輯訪問 (LA) 系統是一個 VGG 網絡和兩個 SincNet 網絡的融合。

## 2. Physical access

2.1. Features and preprocessing 特徵和預處理

For this challenge we explore several features such as Mel-filter bank, MFCC, constant Q-transform (CQT), CQCC, and power spectrogram.

Among the explored features, power spectrogram yields superior performance, followed by CQT features. 

Accordingly, we use these two features in most of our experiments. 

In particular, the submitted systems use either the power spectrograms as a single input channel, or both the power spectrograms and the CQT features fed as two different input channels. 

As a feature preprocessing, both CQT and power spectrogram are first transferred to log domain and then subjected to mean and variance normalization (MVN) before being fed to the network.

對於這個挑戰，該研究探索了幾個特徵，例如梅爾濾波器組(Mel-filter bank)、MFCC、恆定 Q 變換 (constant Q-transform; CQT)、CQCC 和功率譜圖(Power Spectrogram)，在探索的特徵中，功率譜圖產生了優越的性能，其次是 CQT 特徵。該研究在大多數實驗中都使用這兩個特徵。特別是提交的系統使用功率譜圖作為單個輸入通道，或者將功率譜圖和 CQT 特徵作為兩個不同的輸入通道饋送。作為特徵預處理，CQT 和功率譜圖都首先被轉移到對數域，然後在被饋送到網絡之前進行均值和方差歸一化（MVN）。

2.2. Example and minibatch generation for network training 用於網絡訓練的示例和小批量生成

The procedure for generating training examples and minibatches can greatly affect the performance of neural networks in audio processing. 

Therefore, we experimented with several different strategies for this. 

For example generation, we first concatenate all features of the same class (same attack id) and speaker. 

We then split the concatenated features into small segments of the same size.

Initially we used four second segments but after doing several experiments, we found that networks trained on smaller segments performed better than those trained on large segments, mainly because they overfit less to the training data. 

The size of the examples used to train the submitted systems is one second (i.e. 100 frames).

For minibatch generation we experimented with different strategies for distributing the examples into minibatches.

We found that the best strategy is to only use examples from a single speaker within each minibatch (a few minibatches may contain examples from more speakers in order to use all training data). 

Each minibatch has 128 examples. 

After each epoch, we randomise the examples and generate the minibatches again for better generalization.


生成訓練示例和小批量的過程會極大地影響神經網絡在音頻處理中的性能。因此該研究為此嘗試了幾種不同的策略。例如生成研究者首先連接同一類（相同攻擊 id）和說話者的所有特徵，然後將連接的特徵分成相同大小的小段。最初使用了 4 秒的片段，但在進行了幾次實驗後，發現在較小片段上訓練的網絡比在大片段上訓練的網絡表現更好，主要是因為它們對訓練數據的過度擬合較少。用於訓練提交系統的示例大小為一秒（即 100 幀）。對於小批量生成，研究者嘗試了將示例分配到小批量的不同策略。發現最好的策略是在每個 minibatch 中只使用來自單個說話者的示例（一些 minibatch 可能包含來自更多說話者的示例，以便使用所有訓練數據）。每個 minibatch 有 128 個樣本。在每個 epoch 之後，將示例隨機化並再次生成 minibatch 以進行更好的泛化。

2.3. Training and development data 培訓和發展數據

For training the networks, the official training set of the challenge was used. 

This set contains audio samples from 20 speakers. 

One of the speakers was randomly selected for network training validation set which is roughly 5 % of the training data.


The development set is also the official challenge’s development set. 

This set which contains 20 speakers, was only used for evaluating networks and comparing different methods and
training strategies.

為了訓練網絡使用了挑戰的官方訓練集，該集合包含來自 20 個揚聲器的音頻樣本。其中一位發言者被隨機選擇用於網絡訓練驗證集，該驗證集大約是訓練數據的 5%，其開發集也是官方挑戰的開發集，這組包含 20 個揚聲器，僅用於評估網絡和比較不同的方法和訓練策略。

2.4. Networks and training strategies 網絡和培訓策略

For this challenge, two different topologies were used for Physical access.

The first one is a modified version of a VGG network which has shown good performance in Audio Tagging and Audio Scene Classification. 

The second network is a modified version of a Light CNN (LCCN) which had the best performance for ASVSpoof2017 challenge.

We have used a modified version of both networks for acoustic scene classification challenge 2019. 

In the following two sections, both networks will be explained in more detail.

對於這一挑戰，物理訪問使用了兩種不同的拓撲。第一個是 VGG 網絡的修改版本，它在音頻標記和音頻場景分類方面表現出良好的性能，第二個網絡是 Light CNN (LCCN) 的修改版本，在 ASVSpoof2017 挑戰賽中表現最佳。該研究在 2019 年聲學場景分類挑戰賽中使用了這兩個網絡的修改版本。

2.4.1. VGG-like network 類 VGG 網絡

The VGG network comprises several convolutional and pooling layers followed by a statistics pooling and several dense layers which perform classification. 

Table 1 provides a detailed description of the proposed VGG architecture. 

There are 6 convolutional blocks in the model, each containing 2 convolutional layers and one max-pooling. 

Each max-pooling layer reduce the size of frequency axis to half while only one of them reduces the temporal resolution. 

After the convolutional layers, there is a mean pooling layer which operates only on the time axis and calculates the mean over time. 

After this layer, there is a flatten layer which simply concatenates the 4 remaining frequency channels. 

Finally there are 3 dense layers which perform the classification task.

VGG 網絡包括幾個卷積層和池化層，然後是一個統計池和幾個執行分類的密集層。表 1 提供了對所提議的 VGG 架構的詳細描述。模型中有 6 個卷積塊，每個包含 2 個卷積層和一個最大池，每個最大池化層將頻率軸的大小減小到一半，而其中只有一個會降低時間分辨率。在卷積層之後，有一個平均池化層，它只在時間軸上運行併計算隨時間變化的平均值。在這一層之後，有一個扁平層，它簡單地連接剩下的 4 個頻道。最後有 3 個密集層執行分類任務。

2.4.2. Light CNN (LCNN) 輕型 CNN (LCNN)

Table 2 shows the used LCNN topology for this challenge. 

This network is a combination of convolutional and max-pooling layers and uses Max-Feature-Map (MFM) as non-linearity.

MFM is a layer which simply reduce the number of output channels to the half by taking the maximum of two consecutive channels (or any other combination of two channels). 

The rest of this network (statistics and classification parts) is identical to the proposed VGG network.

表 2 顯示了用於該挑戰的 LCNN 拓撲，該網絡是卷積層和最大池化層的組合，並使用 Max-Feature-Map (MFM) 作為非線性。MFM 是一個層，它通過取兩個連續通道的最大值（或兩個通道的任何其他組合）來簡單地將輸出通道的數量減少到一半。該網絡的其餘部分（統計和分類部分）與提議的 VGG 網絡相同。

2.5. Fusion and submitted systems 融合和提交系統

Since the evaluation protocol does not allow us to estimate fusion parameters on the development set, we choose to use a simple average with equal weight for our best systems. 

Our submissions are the following:

- Primary: Fusion of two VGG networks. The first one is trained using two-channels features while the second one is fed with single channel log-power spectrogram.

- Single best: Our single best system for this part is the VGG network with two-channels features.

- Contrastive 1: This system is a VGG network with single channel log-power spectrogram features.

- Contrastive 2: The second contrastive system is LCNN network again with single channel log-power spectrogram as features.


Table 1: The proposed VGG architecture. Conv2D: two dimensional convolutional layer. 

MeanPooling: a layer which calculate the mean in time axis and reduce the shape (remove the time axis). Dense: fully connected dense layer

由於評估協議不允許我們在開發集上估計融合參數，因此該研究選擇對現況最好的系統使用具有相同權重的簡單平均值。

其想法意見如下：

- Primary：兩個 VGG 網絡的融合。 第一個使用雙通道特徵進行訓練，而第二個使用單通道對數功率譜圖進行訓練。
- 單一最佳：我們這部分的單一最佳系統是具有雙通道特徵的 VGG 網絡。
- 對比 1：該系統是一個具有單通道對數功率譜圖特徵的 VGG 網絡。
- 對比2：第二個對比系統是 LCNN 網絡，再次以單通道對數功率譜圖為特徵。


表 1：提議的 VGG 架構。 Conv2D：二維卷積層。MeanPooling：計算時間軸平均值並縮小形狀（移除時間軸）的層。 Dense：全連接密集層

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

該研究展示了 BUT 和 Omilia 聯合提交的 ASVspoof 2019。對於 PA，研究者遵循 VGG 架構，通過僅融合兩個網絡，在開發集和評估集上都獲得了非常有競爭力的結果，對於 LA，該研究將 VGG 架構與最近提出的 SincNet 融合在一起。採用後者的理由是它能夠聯合優化網絡和特徵提取器，這被證明對語音和說話人識別非常有效。儘管我們努力防止過度擬合，類似於主要通過訓練和開發中的攻擊級交叉驗證，但 LA 上的結果顯示 SincNet 難以泛化某些與訓練中的攻擊顯著不同的攻擊。研究者得出的結論是，在訓練和評估攻擊之間存在較大不匹配的情況下，需要進行更多研究才能充分利用 SincNet 等端到端反欺騙架構。

Table 3: Physical access detailed results based on min-tDCF for different conditions. 

The first section shows the baseline results and the second section shows the primary and single best results of the best-performing systems, both from team T28.

Table 4: Logical access detailed results based on min-tDCF for different conditions. 

The first section shows the baseline results and the second section shows the primary system results of the best performing team (T05) as well as the overall best single system results(team T45). 

The bold numbers show conditions where our single system performs better or the same as the best single system.

Table 5: Physical access results of different submissions

Table 6: Logical access results of different submissions

表 3：不同條件下基於 min-tDCF 的物理訪問詳細結果。

第一部分顯示基線結果，第二部分顯示最佳性能係統的主要和單一最佳結果，均來自 T28 團隊。

表 4：不同條件下基於 min-tDCF 的邏輯訪問詳細結果。

第一部分顯示基線結果，第二部分顯示最佳績效團隊 (T05) 的主要係統結果以及整體最佳單系統結果 (團隊 T45)。

粗體數字顯示我們的單一系統表現更好或與最佳單一系統相同的條件。

表 5：不同提交的物理訪問結果

表 6：不同提交的邏輯訪問結果