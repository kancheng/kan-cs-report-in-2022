# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Defending Your Voice: Adversarial Attack on Voice Conversion

https://arxiv.org/abs/2005.08781

Huang, C. Y., Lin, Y. Y., Lee, H. Y., & Lee, L. S. (2021, January). Defending your voice: Adversarial attack on voice conversion. In 2021 IEEE Spoken Language Technology Workshop (SLT) (pp. 552-559). IEEE.

```
Substantial improvements have been achieved in recent years in voice conversion, which converts the speaker characteristics of an utterance into those of another speaker without changing the linguistic content of the utterance. 

Nonetheless, the improved conversion technologies also led to concerns about privacy and authentication.

It thus becomes highly desired to be able to prevent one’s voice from being improperly utilized with such voice conversion technologies.

This is why we report in this paper the first known attempt to perform adversarial attack on voice conversion.

We introduce human imperceptible noise into the utterances of a speaker whose voice is to be defended. 

Given these adversarial examples, voice conversion models cannot convert other utterances so a to sound like being produced by the defended speaker. 

Preliminary experiments were conducted on two currently stateof-the-art zero-shot voice conversion models.

Objective and subjective evaluation results in both white-box and black-box scenarios are reported.

It was shown that the speaker characteristics of the converted utterances were made obviously different from those of the defended speaker, while the adversarial examples of the defended speaker are not distinguishable from the authentic utterances.
```

近年來在語音轉換方面取得了實質性的改進，將一個話語的說話者特徵轉換為另一個說話者的特徵，而不改變話語的語言內容。儘管如此，改進的轉換技術也導致了對隱私和身份驗證的擔憂。因此，非常希望能夠通過這種語音轉換技術來防止一個人的語音被不當使用。這就是為什麼該研究報告了對語音轉換執行對抗性攻擊的第一次已知嘗試，研究者將人類難以察覺的噪音引入說話人的話語中，而說話人的聲音要受到保護。鑑於這些對抗性示例，語音轉換模型無法將其他話語轉換為聽起來像是被防御者發出的聲音。在兩個當前最先進的零樣本語音轉換模型上進行了初步實驗。報告了白盒和黑盒場景中的客觀和主觀評估結果。結果表明，轉換後的話語的說話人特徵與被辯護人的說話人特徵明顯不同，而被辯護人的對抗樣本與真實話語沒有區別。

```
@INPROCEEDINGS{9383529,
  author={Huang, Chien-yu and Lin, Yist Y. and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={Defending Your Voice: Adversarial Attack on Voice Conversion}, 
  year={2021},
  volume={},
  number={},
  pages={552-559},
  doi={10.1109/SLT48900.2021.9383529}}
```

## Code

1. https://yistlin.github.io/attack-vc-demo/

2. https://github.com/cyhuang-tw/attack-vc

3. https://github.com/resemble-ai/Resemblyzer

DEFENDING YOUR VOICE: ADVERSARIAL ATTACK ON VOICE CONVERSION

College of Electrical Engineering and Computer Science, National Taiwan University, Taiwan

ABSTRACT

Substantial improvements have been achieved in recent years in voice conversion, which converts the speaker characteristics of an utterance into those of another speaker without changing the linguistic content of the utterance. 

Nonetheless, the improved conversion technologies also led to concerns about privacy and authentication. It thus becomes highly desired to be able to prevent one’s voice from being improperly utilized with such voice conversion technologies. 

This is why we report in this paper the first known attempt to perform adversarial attack on voice conversion. 

We introduce human imperceptible noise into the utterances of a speaker whose voice is to be defended. 

Given these adversarial examples, voice conversion models cannot convert other utterances so as to sound like being produced by the defended speaker. 

Preliminary experiments were conducted on two currently stateof-the-art zero-shot voice conversion models. 

Objective and subjective evaluation results in both white-box and black-box scenarios are reported. 

It was shown that the speaker characteristics of the converted utterances were made obviously different from those of the defended speaker, while the adversarial examples of the defended speaker are not distinguishable from the authentic utterances.

Index Terms:  voice conversion, adversarial attack, speaker verification, speaker representation

1. INTRODUCTION

Voice conversion aims to alter some specific acoustic characteristics of an utterance, such as the speaker identity, while preserving the linguistic content. 

These technologies were made much more powerful by deep learning [1, 2, 3, 4, 5], but the improved technologies also led to concerns about privacy and authentication.

One’s identity may be counterfeited by voice conversion and exploited in improper ways, which is only one of the many deepfake problems observed today generated by deep learning, such as synthesized fake photos or fake voice. 

Detecting any of such artifacts or defending against such activities is thus increasingly important [6, 7, 8, 9], which applies equally to voice conversion.

On the other hand, it has been widely known that neural networks are fragile in the presence of some specific noise, or prone to yield different or incorrect results if the input is disturbed by such subtle perturbations imperceptible to humans [10]. 

Adversarial attack is to generate such subtle perturbations that can fool the neural networks.

It has been successful on some discriminative models [11, 12, 13], but less reported on generative models [14].

In this paper, we propose to perform adversarial attack on voice conversion to prevent one’s speaker characteristics from being improperly utilized with voice conversion. 

Humanimperceptible perturbations are added to the utterances produced by the speaker to be defended. 

Three different approaches, the end-to-end attack, embedding attack, and feedback attack are proposed, such that the speaker characteristics of the converted utterances would be made very different from those of the defended speaker. 

We conducted objective and subjective evaluations on two recent state-of-the-art zeroshot voice conversion models.

Objective speaker verification showed the converted utterances were significantly different from those produced by the defended speaker, which was then verified by subjective similarity test. 

The effectiveness of the proposed approaches was also verified for black-box attack via a proxy model closer to the real application scenario.

2. RELATED WORKS

2.1. Voice conversion

Traditionally, parallel data are required for voice conversion, or the training utterances of the two speakers must be paired and aligned. 

To overcome this problem, Chou et al. [1] obtained disentangled representations respectively for linguistic content and speaker information with adversarial training; CycleGAN-VC [2] used cycle-consistency to ensure the converted speech to be linguistically meaningful with the target speaker’s features; and StarGAN-VC [3] introduced conditional input for many-to-many voice conversion. 

All these are limited to speakers seen in training.

Zero-shot approaches then tried to convert utterances to any speaker given only one example utterance without finetuning, and the target speaker is not necessarily seen before.

Chou et al. [4] employed adaptive instance normalization for this purpose; AUTOVC [5] integrated a pre-trained d-vector and an encoder bottleneck, achieving the state-of-the-art results.

Fig. 1: The encoder-decoder based voice conversion model and the three proposed approaches. 

Perturbations are updated on the utterances providing speaker characteristics, as the blue dashed lines indicate.

2.2. Attacking and defending voice

Automatic speech recognition (ASR) systems have been shown to be prone to adversarial attacks. 

Applying perturbations on the waveforms, spectrograms, or MFCC features was able to make ASR systems fail to recognize the speech correctly [15, 16, 17, 18, 19]. 

Similar goals were achieved on speaker recognition by generating adversarial examples to fool automatic speaker verification (ASV) systems to predict that these examples had been uttered by a specific speaker [20, 21, 22].

Different approaches for spoofing ASV were also proposed to show the vulnerabilities of such systems [23, 24, 25]. 

But to our knowledge, applying adversarial attacks on voice conversion has not been reported yet.

On the other hand, many approaches were proposed to defend one’s voice when ASV systems were shown to be vulnerable to spoofing attacks [26, 27, 28, 29]. 

In addition to the ASVspoof challenges for spoofing techniques and countermeasures [30], Liu et al. [20] conducted adversarial attacks on those countermeasures, showing the fragility of them. 

Obviously, all neural network models are under the threat of adversarial attacks [11], which led to the idea of attacking voice conversion models as proposed here.


3. METHODOLOGIES

A widely used model for voice conversion adopted an encoderdecoder structure, in which the encoder is further divided into a content encoder and a speaker encoder, as shown in Fig. 1.

This paper is also based on this model. 

The content encoder Ec extracts the content information from an input utterance t yielding Ec(t), while the speaker encoder Es embeds the speaker characteristics of an input utterance x as a latent vector Es(x), as in the left part of Fig. 1. 

Taking Ec(t) and Es(x) as the input, the decoder D generates a spectrogram F(t, x) with content information based on Ec(t) and speaker characteristics based on Es(x).

Here we only focus on the utterances fed into the speaker encoder, since we are defending the speaker characteristics provided by such utterances. 

Motivated by the prior work [14], here we present three approaches to performing the attack, with the target being either the output spectrogram F(t, x) (Sec. 3.1), or the speaker embedding Es(x) (Sec. 3.2), or the combination of the two (Sec. 3.3), as shown in Fig. 1.

3.1. End-to-end attack

3.2. Embedding attack

3.3. Feedback attack

4. EXPERIMENTAL SETTINGS

We conducted experiments on the model proposed by Chou et al. [4] (referred to as Chou’s model below) and AUTOVC.

Both were able to perform zero-shot voice conversion on unseen speakers given their few utterances without finetuning, considered suitable for our scenarios. 

Models such as StarGAN-VC, on the other hand, are limited to the voice produced by speakers seen during training, which makes it less likely to counterfeit other speakers’ voices, and we thus do not consider them here.


4.1. Speaker encoders

For Chou’s model, all modules were trained jointly from scratch on the CSTR VCTK Corpus [32]. 

The speaker encoder took a 512-dim mel spectrogram and generated a 128-dim speaker embedding. 

AUTOVC utilized pre-trained d-vector [33] as speaker encoder, with 80-dim mel spectrogram as input and 256-dim speaker embedding as output, pre-trained on VoxCeleb1 [34] and LibriSpeech [35] but generalizable to unseen speakers.

4.2. Vocoders

In inference, Chou’s model leveraged Griffin-Lim algorithm [36] to synthesize the audio. 

AUTOVC previously adopted WaveNet [37] as the spectrogram inverter, but here we used WaveRNN-based vocoder [38] pre-trained on the VCTK corpus to generate waveforms with similar quality due to time limitation.

As we introduced perturbation on spectrogram, vocoders converting the spectrogram into waveform were necessary. 

We respectively adopted Griffin-lim algorithm and WaveRNN-based vocoder for attacks on Chou’s model and AUTOVC.

4.3. Attack scenarios

Two scenarios were tested here. 

In the first scenario, the attacker has full access to the model to be attacked.

With the complete architecture plus all trained parameters of the model being available, we can apply adversarial attack directly. 

This scenario is referred to as white-box scenario, in which all experiments were conducted on Chou’s model and AUTOVC with their publicly available network parameters and evaluated on exactly the same models.

The second one is referred to as black-box scenario, in which the attacker cannot directly access the parameters of the model to be attacked, or the architecture might even be unknown. 

For attacking Chou’s model, we trained a new model with the same architecture but different initialization, whereas for AUTOVC we trained a new speaker encoder with the architecture similar to the one in the original AUTOVC. 

These newly-trained models are then used as proxy models to generate adversarial examples to be evaluated with the publicly available ones in the same way as in white-box scenario.

Fig. 2: Speaker verification accuracy for the defended speaker with Chou’s ( = 0.075) and AUTOVC ( = 0.05) by the three proposed approaches under the white-box scenario.

4.4. Attack procedure

We selected 2 norm as L(·, ·), and λ = 0.1 for all experiments. 

w in (3), (4), (5) was initialized from a standard normal distribution. 

The perturbations to be added to the utterances was  tanh(w). 

Adam [39] optimizer was adopted to update w iteratively according to the loss function defined in (3), (4), (5) with the learning rate being 0.001 and the number of iterations being 1500.

5. RESULTS

5.1. Objective tests

For automatic evaluation, we adopted speaker verification accuracy as a reliable metric.

The speaker verification system used here first encoded two input utterances into embeddings and then computed the similarity between the two.

The two utterances were considered to be uttered by the same speaker if the similarity exceeds a threshold. 

In the test, each time we compared a machine-generated utterance with a real utterance providing the speaker characteristics in the experiments (x in Fig. 1), and the speaker verification accuracy used below is defined as the percentage of the cases that the two are considered to be produced by the same speaker by the speaker verification system.

The verification system used here was based on a pretrained d-vector model 1 which was different from the speaker encoders of the two attacked models. 

The threshold was determined based on the equal error rate (EER) when verifying utterance pairs randomly sampled from the VCTK corpus in the following way. 

We sampled 256 utterances for each speaker in the dataset, half of which were used as positive samples and the other half as negative ones. 

For positive samples, the similarity was computed with random utterances of the authentic speaker, whereas the negative ones were computed against random utterances produced by other randomly selected speakers. 

This gave the threshold of 0.683 with the EER being 0.056.

We randomly collected enough number of utterances offering the speaker characteristics (x in Fig. 1) from 109 speakers in the VCTK corpus and enough number of utterances offering the content (t in Fig. 1), and performed voice conversion with both Chou’s and AUTOVC to generate the original outputs (F(t, x) in Fig. 1). 


We randomly collected 100 of such pairs (x, F(t, x)) produced by both Chou’s and AUTOVC which were considered by the speaker verification
system mentioned above to be produced by the same speaker
to be used for the test below.2 

Thus the speaker verification accuracy of all these original outputs (F(t, x)) is 1.00. 

We then created corresponding adversarial examples (x + δ in (3), (4), (5)), targeting speakers with gender opposite to the defended speaker, and performed speaker verification respectively on these adversarial example utterances (referred to as adversarial input) and the converted utterances F(t, x + δ) (referred to as adversarial output). 

The same examples were used in the test for Chou’s and AUTOVC.

Fig. 2(a) shows the speaker verification accuracy for the adversarial input and output utterances evaluated with respect to the defended speaker under the white-box scenario for Chou’s model. 

Results for the three approaches mentioned in Sec. 3 are in the three sections (i) (ii) (iii), with the blue crosshatch-dotted bar and the red diagonal-lined bar respectively for adversarial input and adversarial output.

Similarly in Fig. 2(b) for AUTOVC. 

Fig. 3: The speaker verification accuracy for different perturbation  on Chou’s model under black-box scenario for the three proposed approaches: (a) end-to-end, (b) embedding, and (c) feedback attacks.

We can see the adversarial inputs sounded very close to the defended speaker or the perturbation δ almost imperceptible (the blue bars very close to 1.00), while the converted utterances sounded as from a different speaker (the red bars much lower). 

All the three approaches were effective, although feedback attack worked the best for Chou’s (section (iii) in Fig. 2(a)), while embedding attack worked very well for both Chou’s and AUTOVC with respect to both adversarial input and output (section (ii) of each chart).


For black-box scenario, we analyzed the same speaker verification accuracy as in Fig. 2(a) for Chou’s model only but with varying scale of the perturbations , with results plotted in Fig. 3 (a) (b) (c) respectively for the three approaches proposed. 

We see when  = 0.075 the adversarial inputs were kept almost intact (blue curves close to 1.0) while adversarial outputs were seriously disturbed (red curves much lower).

However, as  ≥ 0.1 the speaker characteristics of the adversarial inputs were altered drastically (blue curves went down), although the adversarial outputs sounded very different (red
curves went very low).

Fig. 4 shows the same results as in Fig. 3 except on AUTOVC with embedding attack only (as the other two methods did not work well in white-box scenario in Fig. 2(b)).

We see very similar results as in Fig. 3, and the embedding attack worked successfully with AUTOVC for good choices of (0.05 ≤  ≤ 0.075).

Among the three proposed approaches, the embedding attack turned out to be the most attractive, considering both defending effectiveness (as mentioned above) and time efficiency.

The feedback attack offered very good performance on Chou’s model, but less effective on AUTOVC.

It also took more time to apply the perturbation as one more complete encoder-to-decoder inference was required. 

It is interesting to note that the end-to-end attack offered performance comparable to the other two approaches, although it is based on the distance between spectrograms, very different from the distance between speaker embeddings, based on which the other two approaches relied on.

Fig. 4: Same as Fig. 3 except on AUTOVC and for embedding attack only.

5.2. Subjective tests

The above speaker verification tests were objective but not necessarily adequate.

So we performed subjective evaluation here but only with the most attractive embedding attack on both Chou’s and AUTOVC for both white- and black-box scenarios. 

We randomly selected 50 out of the 100 example utterances (x) from the set used in objective evaluation described above. 

The corresponding adversarial inputs (x + δ), outputs (F(t, x + δ)) and the original outputs (F(t, x)) as used above were then reused in subjective evaluation here for = 0.075 and 0.05 respectively for Chou’s and AUTOVC.

The subjects were then asked to decide if two given utterances were from the same speaker by choosing one out of four: 

(I) Different, absolutely sure,
(II) Different, but not very sure,

(III) Same, but not very sure, and

(IV) Same, absolutely sure. 

Among the two utterances given, one is the original utterance x, whereas the other is the adversarial input, adversarial output, or original output. Each utterance pair was evaluated by 6 subjects.

To remove the possible outliers for sub-jective results, we deleted two extreme ballots on both ends out of the 6 ballots received for each utterance pair (delete a (I) if there is one, or delete a (II) if there is no (I)’s, etc.; similar for (IV) and (III)). 

In this way 4 ballots were collected for each utterance pair, and 200 ballots for the 50 utterance pairs.

The percentages of ballots choosing (I), (II), (III), (IV) out of these 200 are then shown in the bars (1) (2) for white-box, (3) (4) for black-box scenarios, and (5) for original output in

Fig. 5 for Chou’s and AUTOVC respectively.

For Chou’s, we can see in Fig. 5(a) at least 70% - 78% of the ballots chose (IV) or considered the adversarial inputs preserved the original speaker characteristics very well (red parts in bars  (1)(3), yet at least 41% - 58% of the ballots chose (I) or considered adversarial outputs obviously from a different speaker (blue parts in bars (2)(4). 

Also for the original output at least 82% of the ballots considered them close to the original speaker ((III) plus (IV) in bar (5). 

As in Fig. 5(b) for AUTOVC, at least 85% - 90% of the ballots chose (IV) (red parts in bars (1)(3), yet more than 54% - 68% of the ballots chose (I) (blue parts in bars  (2)(4). 

However, only about 27% of the ballots considered that the original outputs are from the same speaker (red and orange parts in bar (5). 

This is probably because the objective speaker verification system used here didn’t match human’s perception very well based on which the selected original output with similarity to the original utterance above the threshold may not sound to human subjects as produced by the same speaker. 

Also for both two models, the black-box scenario was in general more challenging than the white-box one (lower green and blue parts, (4) v.s. (2)), but the approach is still effective to a good extent.

The demo samples can be found at https://yistlin.github.io/attack-vc-demo, and the source code is at https://github.com/cyhuang-tw/attack-vc.