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
