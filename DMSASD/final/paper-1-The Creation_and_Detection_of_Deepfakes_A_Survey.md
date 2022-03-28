# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

0. The Creation and Detection of Deepfakes: A Survey

https://arxiv.org/abs/2004.11138


```
Generative deep learning algorithms have progressed to a point where it is difficult to tell the difference between what is real and what is fake. 

In 2018, it was discovered how easy it is to use this technology for unethical and malicious applications, such as the spread of misinformation, impersonation of political leaders, and the defamation of innocent individuals. Since then, these 'deepfakes' have advanced significantly.

In this paper, we explore the creation and detection of deepfakes and provide an in-depth view of how these architectures work.

The purpose of this survey is to provide the reader with a deeper understanding of (1) how deepfakes are created and detected, (2) the current trends and advancements in this domain, (3) the shortcomings of the current defense solutions, and (4) the areas which require further research and attention.
```

目前生成式深度學習算法已經發展到難以區分真假的程度，而自 2018 年起，人們發現將這項技術用於不道德和惡意應用是多麼容易，例如傳播錯誤信息、冒充政治領導人以及誹謗無辜個人。 從那時起，這些“深度偽造”取得了顯著進展。該研究探討了 deepfakes 的創建和檢測，並深入了解這些架構的工作原理。本次調查的目的是讓讀者更深入地了解（1）如何創建和檢測深度偽造，（2）該領域的當前趨勢和進步，（3）當前防禦解決方案的缺點，以及（ 4）需要進一步研究和關注的領域。

Bibliography

```
@article{DBLP:journals/corr/abs-2004-11138,
  author    = {Yisroel Mirsky and
               Wenke Lee},
  title     = {The Creation and Detection of Deepfakes: {A} Survey},
  journal   = {CoRR},
  volume    = {abs/2004.11138},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.11138},
  eprinttype = {arXiv},
  eprint    = {2004.11138},
  timestamp = {Tue, 28 Apr 2020 16:10:02 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-11138.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

A deepfake is content, generated by an artificial intelligence, that is authentic in the eyes of a human being.

The word deepfake is a combination of the words ‘deep learning’ and ‘fake’ and primarily relates to content generated by an artificial neural network, a branch of machine learning.

The most common form of deepfakes involve the generation and manipulation of human imagery. 

This technology has creative and productive applications. For example, realistic video dubbing of foreign films, education though the reanimation of historical figures, and virtually trying on clothes while shopping. 

There are also numerous online communities devoted to creating deepfake memes for entertainment, such as music videos portraying the face of actor Nicolas Cage.

However, despite the positive applications of deepfakes, the technology is infamous for its unethical and malicious aspects. 

At the end of 2017, a Reddit user by the name of ‘deepfakes’ was using deep learning to swap faces of celebrities into pornographic videos, and was posting them online. 

The discovery caused a media frenzy and a large number of new deepfake videos began to emerge thereafter.

In 2018, BuzzFeed released a deepfake video of former president Barak Obama giving a talk on the subject. 

The video was made using the Reddit user’s software (FakeApp), and raised concerns over identity theft, impersonation, and the spread of misinformation on social media.

Following these events, the subject of deepfakes gained traction in the academic community, and the technology has
been rapidly advancing over the last few years. Since 2017,

the number of papers published on the subject rose from 3 to over 150 (2018-19).

To understand where the threats are moving and how to mitigate them, we need a clear view of the technology’s, challenges, limitations, capabilities, and trajectory. Unfortunately,
to the best of our knowledge, there are no other works which present the techniques, advancements, and challenges, in a technical and encompassing way. 

Therefore, the goals of this paper are (1) to provide the reader with an understanding of how modern deepfakes are created and detected, (2) to inform the reader of the recent advances, trends, and challenges in deepfake research, (3) to serve as a guide to the design of deepfake architectures, and (4) to identify the current status of the attacker-defender game, the attacker’s next move, and future work that may help give the defender a leading edge.

We achieve these goals through an overview of human visual deepfakes (Section II), followed by a technical background
which identifies technology’s basic building blocks, limitations, and challenges (Section III). 

We then provide a chronological and systematic review for each category of deepfake, and provide the networks’ schematics to give the reader a deeper understanding of the various approaches (Sections IV and V).

Finally, after reviewing the countermeasures (Section VI), we discuss their weaknesses, suggest alternative research, consider the adversary’s next steps, and raise awareness to the spread of deepfakes to other domains (Section VII).

Fig. 1: The difference between adversarial machine learning and deepfakes.

Scope. 

In this SoK we will focus on deepfakes pertaining to the human face and body. We will not be discussing the synthesis of new faces or the enhancement of facial features because they do not have a clear attack goal associated with them.

In Section VII-C we will discuss deepfakes with a much broader scope, note the future trends, and exemplify how deepfakes
have spread to other domains and media such as forensics, finance, and healthcare. 

We note to the reader that deepfakes should not be confused with adversarial machine learning, which is the subject of fooling machine learning algorithms with maliciously crafted inputs (Fig. 1). 

The difference being that for deepfakes, the objective of the generated content is to fool a human and not a machine.


Deepfake 是由人工智能生成的內容，在人類眼中是真實的。deepfake 這個詞是“深度學習”和“假”兩個詞的組合，主要與人工神經網絡（機器學習的一個分支）生成的內容有關。最常見的深度偽造形式涉及人類圖像的生成和操縱。該技術具有創造性和生產性的應用。例如，外國電影的真實視頻配音，通過歷史人物的複活進行教育，以及在購物時虛擬試穿衣服。

還有許多在線社區致力於為娛樂創建 deepfake 模因，例如描繪演員 Nicolas Cage 面孔的音樂視頻。然而，儘管深度偽造的積極應用，該技術因其不道德和惡意方面而臭名昭著。

2017 年底，一位名為“deepfakes”的 Reddit 用戶正在使用深度學習將名人的面孔轉換為色情視頻，並將其發佈到網上。這一發現引起了媒體的狂熱，此後大量新的 deepfake 視頻開始出現。2018 年，BuzzFeed 發布了一段前總統巴拉克·奧巴馬 (Barak Obama) 就該主題發表演講的 deepfake 視頻。該視頻是使用 Reddit 用戶的軟件 (FakeApp) 製作的，引發了對身份盜用、假冒和社交媒體上錯誤信息傳播的擔憂。

在這些事件之後，deepfakes 的主題在學術界引起了關注，該技術已經
近幾年發展迅速。自 2017 年以來，在該主題上發表的論文數量從 3 篇增加到 150 多篇（2018-19 年）。要了解威脅的移動方向以及如何減輕威脅，我們需要清楚地了解技術、挑戰、限制、能力和軌跡。很遺憾，
據我們所知，沒有其他作品以技術性和包容性的方式呈現技術、進步和挑戰。

因此，本文的目標是 (1) 讓讀者了解如何創建和檢測現代 deepfake，(2) 讓讀者了解 deepfake 研究的最新進展、趨勢和挑戰，(3)作為 deepfake 架構設計的指南，以及 (4) 確定攻擊者與防御者博弈的當前狀態、攻擊者的下一步行動以及可能有助於讓防御者處於領先地位的未來工作。

我們通過對人類視覺深度偽造的概述（第二部分）以及技術背景來實現這些目標，它確定了技術的基本組成部分、限制和挑戰（第 III 節）。
然後，我們為每個 deepfake 類別提供按時間順序和系統的回顧，並提供網絡示意圖，讓讀者更深入地了解各種方法（第 IV 和 V 節）。
最後，在回顧了對策（第 VI 節）之後，我們討論了它們的弱點，提出了替代研究建議，考慮了對手的下一步行動，並提高了對 deepfake 向其他領域傳播的認識（第 VII 節）。

圖 1：對抗性機器學習和 deepfakes 之間的區別。

範圍。

在這個 SoK 中，我們將重點關注與人臉和身體有關的 deepfakes。我們不會討論新面孔的合成或面部特徵的增強，因為它們沒有與之相關的明確攻擊目標。在第 VII-C 節中，我們將討論範圍更廣的 deepfakes，注意未來趨勢，並舉例說明 deepfakes 如何
已經傳播到其他領域和媒體，例如取證、金融和醫療保健。我們提醒讀者，深度偽造不應與對抗性機器學習相混淆，對抗性機器學習是用惡意製作的輸入來欺騙機器學習算法的主題（圖 1）。不同之處在於，對於 deepfakes，生成內容的目的是欺騙人類而不是機器。


II. OVERVIEW & ATTACK MODELS

$x_s$ 操作 $x_t$ 的一舉一動

We identify four categories of deepfakes relating to human visuals: reenactment, replacement, editing, and synthesis.

Although human image editing and synthesis are active research topics, reenactment and replacement deepfakes are the greatest concern because they give an attacker control over one’s identity. 

Fig. 2 illustrates some examples of these four categories and their sub-types in the context of faces.

Throughout this paper we denote s and t as the source and the target identities. 

We also denote $x_s$ and $x_t$ as images of these identities and $x_g$ as the deepfake generated from s and t.

A. Reenactment

A reenactment deepfake is where xs is used to drive the expression, mouth, gaze, pose, or body of xt:

Expression reenactment is where xs drives the expression of xt. 

It is the most common form of reenactment since these technologies often drive target’s mouth and pose as well, providing a wide range of flexibility. 

Benign uses are found in the movie and video game industry where the performances of actors are tweaked in post, and in educational media where historical figures are reenacted.

Mouth reenactment, also known as ‘dubbing’, is where the mouth of xt is driven by that of xs, or an audio input as containing speech. 

Benign uses of the technology includes realistic voice dubbing into another language and editing.

Gaze reenactment is where direction of xt’s eyes, and the position of the eyelids, are driven by those of xs. 

This is used to improve photographs or to automatically maintain eye contact during video interviews.

Pose reenactment is where the head position of xt is driven by xs. 

This technology has primarily been used for face frontalization of individuals in security footage, and as a means for improving facial recognition software.

Body reenactment, a.k.a. pose transfer and human pose synthesis, is similar to the facial reenactments listed above except that’s its the pose of xt’s body being driven.

The Attack Model. 

Reenactment deep fakes give attackers the ability to impersonate an identity, controlling what he or she says or does. 

This enables an attacker to perform acts of defamation, cause discredability, spread misinformation, and tamper with evidence.

For example, an attacker can impersonate t to gain trust the of a colleague, friend, or family member as a means to gain access to money, network infrastructure, or some other asset. 

An attacker can also generate embarrassing content of t for blackmailing purposes or generate content to affect the public’s opinion of an individual or political leader.

Finally, the technology can be used to tamper surveillance footage or some other archival imagery in an attempt to plant false evidence in a trial.

我們確定了與人類視覺相關的四類深度偽造：重演、替換、編輯和合成。

儘管人類圖像編輯和合成是活躍的研究課題，但重演和替換 deepfakes 是最受關注的問題，因為它們使攻擊者能夠控制自己的身份。

圖 2 說明了這四個類別及其子類型在人臉上下文中的一些示例。

在整篇論文中，我們將 s 和 t 表示為源身份和目標身份。

我們還將 $x_s$ 和 $x_t$ 表示為這些身份的圖像，並將 $x_g$ 表示為從 s 和 t 生成的 deepfake。

A. 重演

重演 deepfake 是 xs 用於驅動 xt 的表情、嘴巴、凝視、姿勢或身體的地方：

表達式重演是 xs 驅動 xt 表達式的地方。

這是最常見的重演形式，因為這些技術通常也會驅動目標的嘴巴和姿勢，提供廣泛的靈活性。

在電影和視頻遊戲行業中發現了良性用途，在這些行業中演員的表演在後期進行了調整，並在教育媒體中重演了歷史人物。

嘴巴重演，也稱為“配音”，是 xt 的嘴巴由 xs 的嘴巴驅動的地方，或者是包含語音的音頻輸入。該技術的良性用途包括將逼真的語音配音成另一種語言和編輯。

凝視重演是 xt 眼睛的方向和眼瞼的位置由 xs 的驅動。這用於改善照片或在視頻採訪期間自動保持眼神交流。

姿勢重演是 xt 的頭部位置由 xs 驅動。該技術主要用於安全鏡頭中個人的面部正面化，並作為改進面部識別軟件的一種手段。

身體重演，也稱為姿勢轉移和人體姿勢合成，類似於上面列出的面部重演，只是它是驅動 xt 身體的姿勢。

攻擊模型。

重演深度偽造使攻擊者能夠冒充身份，控制他或她所說或所做的事情。這使攻擊者能夠執行誹謗行為、造成不可信、傳播錯誤信息和篡改證據。例如，攻擊者可以冒充 t 來獲得同事、朋友或家人的信任，以此作為獲取金錢、網絡基礎設施或其他資產的手段。攻擊者還可以出於勒索目的生成令人尷尬的 t 內容，或生成影響公眾對個人或政治領袖的看法的內容。

最後，該技術可用於篡改監控錄像或其他一些檔案圖像，以試圖在審判中植入虛假證據。


B. Replacement

A replacement deepfake is where the content of xt is replaced with that of xs, preserving the identity of s.

Transfer is where the content of xt is replaced with that of xs.

A common type of transfer is facial transfer, used in the fashion industry to visualize an individual in different outfits.

Swap is where the content transferred to xt from xs is driven by xt. 

The most popular type of swap replacement is ‘face swap’, often used to generate memes or satirical content by swapping the identity of an actor with that of a famous individual. 

Another benign use for face swapping includes the anonymization of one’s identity in public content in-place of blurring or pixelation.

The Attack Model. 

Replacement deepfakes are well-known for their harmful applications. 

For example, revenge porn is where an attacker swaps a victim’s face onto the body of a porn actress to humiliate, defame, and blackmail the victim. 

Face replacement can also be used as a short-cut to fully reenacting t by transferring t’s face onto the body of a look-alike. 

This approach has been used as a tool for disseminating political opinions in the past.

B. 更換

替換 deepfake 是將 $x_t$ 的內容替換為 $x_s$ 的內容，保留 s 的身份。轉移是將 $x_t$ 的內容替換為 $x_s$ 的內容。一種常見的轉移類型是面部轉移，在時尚行業中用於形象化穿著不同服裝的個人。

交換是從 $x_s$ 傳輸到 $x_t$ 的內容由 $x_t$ 驅動的地方。最流行的交換替換類型是“面部交換”，通常用於通過將演員的身份與名人的身份交換來生成模因或諷刺內容。

換臉的另一個良性用途包括在公共內容中匿名化一個人的身份，而不是模糊或像素化。

攻擊模型。

替換 deepfakes 以其有害應用而聞名。例如，復仇色情是攻擊者將受害者的臉換到色情女演員的身上，以羞辱、誹謗和勒索受害者。

換臉也可以用作通過將 t 的臉轉移到相似的身體上來完全重現 t 的捷徑。這種方法過去曾被用作傳播政治觀點的工具。

C. Enhancement

An enchantment deepfake is where the attributes of xt
are added, altered, or removed.

Some examples include the changing a target’s clothes, facial hair, age, weight, beauty, and ethnicity. 

Apps such as FaceApp enable users to alter their appearance for entertainment and easy editing of multimedia.

The same process can be used by and attacker to build a false persona for misleading others. 

For example, a sick leader can be made to look healthy, and child or sex predators can change their age and gender to build dynamic profiles online. 

A known unethical use of enhancement deepfakes is the removal of a victim’s clothes for humiliation or entertainment.

D. Synthesis

Synthesis is where the deepfake xg is created with no target as a basis.

Human face and body synthesis techniques such as (used in Fig. 2) can create royalty free stock footage or
generate characters for movies and games. 

However, similar to enhancement deepfakes, it can also be used to create fake personas online.

C. 增強

一個結界 deepfake 是 xt 的屬性所在被添加、更改或刪除。一些例子包括改變目標的衣服、面部毛髮、年齡、體重、美麗和種族。

FaceApp 等應用程序使用戶能夠改變他們的外觀以進行娛樂和輕鬆編輯多媒體。攻擊者可以使用相同的過程來建立虛假角色以誤導他人。例如，可以讓生病的領導者看起來很健康，而兒童或性侵犯者可以改變他們的年齡和性別來建立在線動態檔案。增強深度偽造的一個已知不道德使用是為了羞辱或娛樂而脫掉受害者的衣服。

D. 合成

合成是在沒有目標作為基礎的情況下創建 deepfake $x_g$ 的地方。人臉和身體合成技術，例如（在圖 2 中使用）可以創建免版稅影視素材或為電影和遊戲生成角色。但是，與增強 deepfakes 類似，它也可用於在線創建虛假角色。

Fig. 2: Examples and illustrations of reenactment, replacement, editing, and synthesis deepfakes of the human face.

Fig. 3: Five basic neural network architectures used to create deepfakes.

The lines indicate dataflows used during deployment (black) and training (grey).

TABLE I: Summary of Deep Learning Reenactment Models (Body and Face)

TABLE II: Summary of Deep Learning Replacement Models

Fig. 4: Architectural schematics of reenactment networks. Black lines indicate prediction flows used during deployment, dashed gray lines indicate dataflows performed during training. Zoom in for more detail.

Fig. 5: Architectural schematics of reenactment networks. Black lines indicate prediction flows used during deployment, dashed gray lines indicate dataflows performed during training. Zoom in for more detail.

Fig. 6: Architectural schematics of the reenactment networks. 

Black lines indicate prediction flows used during deployment, dashed gray lines indicate dataflows performed during training. Zoom in for more detail.

Fig. 7: Architectural schematics of the replacement networks. Black lines indicate prediction flows used during deployment, dashed gray lines indicate dataflows performed during training. Zoom in for more detail.

TABLE II: Summary of Deep Learning Replacement Models

TABLE III: Summary of Deepfake Detection Models

圖 2：人臉的重演、替換、編輯和合成 deepfakes 的示例和插圖。

圖 3：用於創建 deepfake 的五種基本神經網絡架構。這些線表示部署（黑色）和訓練（灰色）期間使用的數據流。

表一：深度學習重演模型總結（身體和臉部）

表二：深度學習替代模型總結

圖 4：重演網絡的架構示意圖。黑線表示部署期間使用的預測流，灰色虛線表示訓練期間執行的數據流。放大以獲得更多細節。

圖 5：重演網絡的架構示意圖。黑線表示部署期間使用的預測流，灰色虛線表示訓練期間執行的數據流。放大以獲得更多細節。

圖 6：重演網絡的架構示意圖。

黑線表示部署期間使用的預測流，灰色虛線表示訓練期間執行的數據流。放大以獲得更多細節。

圖 7：替換網絡的架構示意圖。黑線表示部署期間使用的預測流，灰色虛線表示訓練期間執行的數據流。放大以獲得更多細節。

表二：深度學習替代模型總結

表三：Deepfake 檢測模型總結

III. TECHNICAL BACKGROUND

Although there are a wide variety of neural networks, most deepfakes are created using variations or combinations of generative networks and encoder decoder networks. 

In this section we provide a brief introduction to these networks, how they are trained, and the notations which we will be using throughout the paper.

儘管神經網絡種類繁多，但大多數深度偽造都是使用生成網絡和編碼器解碼器網絡的變體或組合創建的。在本節中，我們將簡要介紹這些網絡、它們是如何訓練的，以及我們將在整篇論文中使用的符號。


A. Neural Networks

B. Loss Functions

C. Generative Neural Networks (for deepfakes)

Deep fakes are often created using combinations or variations of six different networks, five of which are illustrated in Fig. 3.

Encoder-Decoder Networks (ED). 

An ED consists of at least two networks, an encoder En and decoder De.

The ED has narrower layers towards its center so that when it’s trained as De(En(x)) = xg, the network is forced to summarize the observed concepts.

The summary of x, given its distribution X, is En(x) = e, often referred to as an encoding or embedding and E = En(X) is referred to as the ‘latent space’. 

Deepfake technologies often use multiple encoders or decoders and manipulate the encodings to influence the output xg. If an encoder and decoder are symmetrical, and the network is trained with the objective De(En(x)) = x, then the network is called an autoencoder and the output is the reconstruction of x denoted xˆ.

Another special kind of ED is the variational autorencoder (VAE) where the encoder learns the posterior distribution of the decoder given X. 

VAEs are better at generating content than autoencoders because the concepts in the latent space are disentangled, and thus encodings respond better to interpolation and modification.

深度偽造通常是使用六個不同網絡的組合或變體創建的，其中五個如圖 3 所示。

編碼器-解碼器網絡 (ED)。

一個 ED 至少由兩個網絡組成，一個編碼器 En 和一個解碼器 De。ED 的中心層較窄，因此當它被訓練為 De(En(x)) = xg 時，網絡被迫總結觀察到的概念。給定分佈 X，x 的摘要是 En(x) = e，通常被稱為編碼或嵌入，而 E = En(X) 被稱為“潛在空間”。

Deepfake 技術通常使用多個編碼器或解碼器並操縱編碼以影響輸出 $x_g$。如果編碼器和解碼器是對稱的，並且使用目標 De(En(x)) = x 訓練網絡，則該網絡稱為自編碼器，輸出是 x 的重建，表示為 $\hat{\boldsymbol{x}}$ 。另一種特殊的 ED 是變分自動編碼器 (VAE)，其中編碼器學習給定 X 的解碼器的後驗分佈。VAE 比自動編碼器更擅長生成內容，因為潛在空間中的概念被解開，因此編碼對插值和修改的響應更好。

Convolutional Neural Network (CNN).

In contrast to a fully connected (dense) network, a CNN learns pattern hierarchies in the data and is therefore much more efficient at handling imagery.

A convolutional layer in a CNN learns filters which are shifted over the input forming an abstract feature map as the output. 

Pooling layers are used to reduce the dimensionality as the network gets deeper and up-sampling layers are used to increase it.

With convolutional, pooling, and upsampling layers, it is possible to build an ED CNNs for imagery.

Generative Adversarial Networks (GAN)

The GAN was first proposed in 2014 by Goodfellow et al. in [11]. 

A GANs consist of two neural networks which work against each other:

the generator G and the discriminator D.

G creates fake samples xg with the aim of fooling D, and D learns to differentiate between real samples (x ∈ X) and fake samples ($x_g$ = G(z) where z ∼ N). 

Concretely, there is an adversarial loss used to train D and G respectively:

$$
\begin{gathered}
\mathcal{L}_{a d v}(D)=\max \log D(x)+\log (1-D(G(z))) \\
\mathcal{L}_{a d v}(G)=\min \log (1 D(G(z)))
\end{gathered}
$$

卷積神經網絡 (CNN)。

與全連接（密集）網絡相比，CNN 學習數據中的模式層次結構，因此在處理圖像方面效率更高。CNN 中的捲積層學習過濾器，這些過濾器在輸入上移動，形成抽象特徵圖作為輸出。當網絡變得更深時，池化層用於降低維數，而上採樣層用於增加它。通過卷積、池化和上採樣層，可以為圖像構建 ED CNN。

生成對抗網絡 (GAN)

GAN 於 2014 年由 Goodfellow 等人首次提出。GAN 由兩個相互對抗的神經網絡組成：

生成器 G 和判別器 D。

G 創建假樣本 xg 的目的是欺騙 D，並且 D 學會區分真實樣本 (x ∈ X) 和假樣本 ($x_g$ = G(z)，其中 z ∼ N)。具體來說，有一個對抗性損失用於分別訓練 D 和 G：


This zero-sum game leads to G learning how to generate samples that are indistinguishable from the original distribution. 

After training, D is discarded and G is used to generate content.

When applied to imagery, this approach produces photo realistic images.

Image-to-Image Translation (pix2pix). 

Numerous of variations and improvements on GANs have been proposed over the years. 

One popular version is the pix2pix framework which enables translations from one image domain to another. 

In pix2pix, G tries to generate the image xg given a visual context xc as an input, and D discriminates between (x, $x_c$) and ($x_g$, $x_c$).

Moreover, G is a an ED CNN with skip connections from En to De (called a U-Net) which enables G to produce high fidelity imagery by bypassing the compression layers when needed.

Later, pix2pixHD was proposed for generating high resolution imagery with better fidelity.

CycleGAN. 

An improvement of pix2pix which enables image translation through unpaired training.

The network forms a cycle consisting of two GANs used to convert images from one domain to another, and then back again to ensure consistency with a cycle consistency loss (Lcyc).

Recurrent Neural Networks (RNN) 

An RNN is type of neural network that can handle sequential and variable length data. 

The network remembers is internal state after processing x(i − 1) and can use it to process x(i) and so on. 

In deepfake creation, RNNs are often used to handle audio and sometimes video. 

More advanced versions of RNNs include long shortterm memory (LSTM) and gate reccurent units (GRU).

這種零和遊戲導致 G 學習如何生成與原始分佈無法區分的樣本，訓練後丟棄 D，使用 G 生成內容，當應用於圖像時，這種方法會產生照片般逼真的圖像。

Image-to-Image Translation (pix2pix)。

多年來，人們提出了對 GAN 的許多變化和改進。一個流行的版本是 pix2pix 框架，它可以實現從一個圖像域到另一個域的轉換。在 pix2pix 中，G 嘗試在給定視覺上下文 xc 作為輸入的情況下生成圖像 xg，並且 D 區分 (x, $x_c$) 和 ($x_g$, $x_c$)。此外，G 是一個 ED CNN，具有從 En 到 De 的跳躍連接（稱為 U-Net），它使 G 能夠在需要時繞過壓縮層來生成高保真圖像。後來，pix2pixHD 被提出用於生成具有更好保真度的高分辨率圖像。

CycleGAN

pix2pix 的改進，可通過非配對訓練實現圖像翻譯。該網絡形成一個由兩個 GAN 組成的循環，用於將圖像從一個域轉換到另一個域，然後再返回以確保與循環一致性損失 (Lcyc) 的一致性。

Recurrent Neural Networks (RNN)

RNN 是一種可以處理順序和可變長度數據的神經網絡。網絡在處理 x(i - 1) 後會記住內部狀態，並可以使用它來處理 x(i) 等等。在 deepfake 創建中，RNN 通常用於處理音頻，有時也用於處理視頻，更外高級的 RNN 版本包括長短期記憶 (LSTM) 和門循環單元 (GRU)。

D. Feature Representations

E. Deepfake Creation Basics

F. Generalization

G. Challenges

- Generalization.

- Paired Training.

- Identity Leakage.

- Occlusions.

- Temporal Coherence.


IV. REENACTMENT

In this section we present a chronological review of deep learning based reenactment, organized according to their class of identity generalization. 

Table I provides a summary and systematization of all the works mentioned in this section.

A. Expression Reenactment

Expression reenactment turns an identity into a puppet, giving attackers the most flexibility to achieve their desired impact.

Before we review the subject, we note that expression reenactment has been around long before deepfakes were popularized. 

In 2003, researchers morphed models of 3D scanned heads. 

In 2005, it was shown how this can be done without a 3D model. 

Later, between 2015 and 2018, 

Thies et al. demonstrated how 3D parametric models can be used to achieve high quality and real-time results with depth sensing and ordinary cameras.

Regardless, today deep learning approaches are recognized as the simplest way to generate believable content. 

To help the reader understand the networks and follow the text, we provide network schematics and their loss functions in Fig. 4.

在本節中，我們將按時間順序回顧基於深度學習的重演，根據他們的身份泛化類別進行組織。表 I 提供了本節中提到的所有工作的總結和系統化。

A. 表情重演

表情重演將身份變成傀儡，為攻擊者提供最大的靈活性來實現他們想要的影響。在我們回顧這個主題之前，我們注意到表情重演早在深度偽造普及之前就已經存在了。2003 年，研究人員對 3D 掃描頭部模型進行了變形。2005 年，展示瞭如何在沒有 3D 模型的情況下完成此操作。後來，在 2015 年至 2018 年間，Thies et al. 演示瞭如何使用 3D 參數模型通過深度傳感和普通相機實現高質量和實時結果（[54] 和 [55]、[56]）。
無論如何，今天的深度學習方法被認為是生成可信內容的最簡單方法。為了幫助讀者理解網絡並遵循文本，我們在圖 4 中提供了網絡示意圖及其損失函數。

1) One-to-One (Identity to Identity):

In 2017, the auhtors of [57] proposed using a CycleGAN for facial reenactment, without the need for data pairing. 

The two domains where video frames of s and t. 

However, to avoid artifacts in xg, the authors note that both domains must share a similar distributions (e.g., poses and expressions). 

In 2018, Bansal et al. proposed a generic translation network based on CycleGAN called Recycle-GAN [58]. 

Their framework improves temporal coherence and mitigates artifacts by including next-frame predictor networks for each domain. 

For facial reenactment, the authors train their network to translate the facial landmarks of xs into portraits of xt.

一對一 (One-to-One, Identity to Identity)：

Runze Xu 等人 CycleGAN 進行臉部重演，無需數據配對，s 和 t 的視頻幀所在的兩個域，為了避免 $x_g$ 中的偽影，研究者指出兩個域必須如姿勢和表情等共享相似的分佈。另外在 2018 年時由 Bansal 等人提出了一個基於 CycleGAN 的通用翻譯網絡，稱為 Recycle-GAN，其框架通過包含每個域的下一幀預測器網絡來提高時間一致性並減輕偽影。對於臉部重演，研究者者訓練他們的網絡將 $x_s$ 的面部標誌轉換為 $x_t$ 的肖像。

2) Many-to-One (Multiple Identities to a Single Identity):

In 2017, the authors of [59] proposed CVAE-GAN, a conditional VAE-GAN where the generator is conditioned on an attribute vector or class label. 

However, reenactment with CVAE-GAN requires manual attribute morphing by interpolating the latent variables (e.g., between target poses).

Later, in 2018, a large number of source-identity agnostic models were published, each proposing a different method to decoupling s from t:

Facial Boundary Conversion. One approach was to first convert the structure of source’s facial boundaries to that of the target’s before passing them through the generator [19].

Their framework ‘ReenactGAN’ the authors use a CycleGAN to transform the boundary bs to the target’s face shape as bt before generating xg with a pix2pix-like generator.

多對一 (Many-to-One, Multiple Identities to a Single Identity)：

2017 年，Jianmin Bao 等人提出了 CVAE-GAN，這是一種條件 VAE-GAN，其中生成器以屬性向量或類標籤為條件，然而使用 CVAE-GAN 重新制定需要在類似於在目標姿勢之間通過插值潛在變量進行手動屬性變形，後來在 2018 年時，發布了大量與源身份無關的模型，其每個模型都提出了一種不同的方法來將 s 與 t 進行解耦，而所謂的面部邊界轉換，是一種方法是首先將源面部邊界的結構轉換為目標的面部邊界結構，然後再將它們傳遞給生成器。如框架 "ReenactGAN" 的研究者使用 CycleGAN 將邊界 $b_s$ 轉換為目標的面部形狀作為 $b_t$ ，然後使用類似 pix2pix 的生成器生成 $x_g$。

...

3) Many-to-Many (Multiple IDs to Multiple IDs): 

The first attempts at identity agnostic models were made in 2017, where the authors of [60] used a conditional GAN (CGAN) for the task. 

Their approach was to (1) extract the inner-face regions as (xt, xs), and then (2) pass them an ED to produce xg subjected to L1 and Ladv losses. 

The challenge of using a CGAN was that the training data had to be paired (images of different identities with the same expression). 

Going one step further, in [61] the authors reenacted full portraits at low resolutions. 

Their approach was to decoupling the identities was to use a conditional adversarial autoencoder to disentangle the identity from the expression in the latent space.

However, their approach is limited to driving xt with discreet AU expression labels (fixed expressions) that capture xs.

多對多 (Many-to-Many, Multiple IDs to Multiple IDs)：

身份不可知模型的首次嘗試是在 2017 年，Kyle Olszewski 等人使用 conditional GAN（CGAN）來完成任務，其方法是 (1) 將內面區域提取為 ($x_t$, $x_s$)，然後 (2) 將它們傳遞給 ED 以產生 $x_g$ 受到 $L_1$ 和 $L_{adv}$ 損失，同時使用 CGAN 的挑戰在於必須對訓練數據進行配對，比如具有相同表情的不同身份的圖像。更進一步來說，在 Yuqian Zhou 等人的工作中，作者以低分辨率重新製作了完整的肖像，他們的方法是將身份解耦，即使用條件對抗自動編碼器將身份與潛在空間中的表達式分離。然而，他們的方法僅限於使用捕獲 $x_s$ 的謹慎 AU 表達式標籤（固定表達式）來驅動 $x_t$。


VIII. CONCLUSION

Not all deepfakes are malicious. 

However, because the technology makes it so easy to create believable media, malicious users are exploiting it to perform attacks. 

These attacks are targeting individuals and causing psychological, political, monetary, and physical harm. 

As time goes on, we expect to see these malicious deepfakes spread to many other modalities and industries.

In this SoK we focused on reenactment and replacement deepfakes of humans.

We provided a deep review of how these technologies work, the differences between their architectures, and what is being done to detect them. 

We hope this information will be helpful to the community in understanding and preventing malicious deepfakes.

并非所有深度伪造都是恶意的。然而，由于该技术使得创建可信媒体变得如此容易，恶意用户正在利用它进行攻击。

这些攻击针对个人并造成心理、政治、金钱和身体伤害。随着时间的推移，我们希望看到这些恶意深度伪造传播到许多其他模式和行业。

在这个 SoK 中，我们专注于人类的重演和替换 deepfakes。我们深入回顾了这些技术的工作原理、它们的架构之间的差异，以及正在采取哪些措施来检测它们。我们希望这些信息有助于社区了解和防止恶意深度伪造。