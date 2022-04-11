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