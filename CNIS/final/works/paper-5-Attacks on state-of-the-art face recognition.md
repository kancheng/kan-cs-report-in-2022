# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Song Q, Wu Y, Yang L. Attacks on state-of-the-art face recognition using attentional adversarial attack generative network. arXiv preprint arXiv:1811.12026, 2018.

Yang, L., Song, Q., & Wu, Y. (2021). Attacks on state-of-the-art face recognition using attentional adversarial attack generative network. Multimedia Tools and Applications, 80(1), 855-875.

Link : https://arxiv.org/abs/1811.12026

Note : 使用注意力机制和生成对抗网络生成指定语义信息的假人脸,使得人脸识别器误判

```
With the broad use of face recognition, its weakness gradually emerges that it is able to be attacked.

So, it is important to study how face recognition networks are subject to attacks.

In this paper, we focus on a novel way to do attacks against face recognition network that misleads the network to identify someone as the target person not misclassify inconspicuously.

Simultaneously, for this purpose, we introduce a specific attentional adversarial attack generative network to generate fake face images.

For capturing the semantic information of the target person, this work adds a conditional variational autoencoder and attention modules to learn the instance-level correspondences between faces.

Unlike traditional two-player GAN, this work introduces face recognition networks as the third player to participate in the competition between generator and discriminator which allows the attacker to impersonate the target person better.

The generated faces which are hard to arouse the notice of onlookers can evade recognition by state-of-the-art networks and most of them are recognized as the target person.
```

隨著人臉識別的廣泛應用，它的弱點也逐漸暴露出來，即容易被攻擊。因此研究人臉識別網絡如何受到攻擊非常重要。在研究中，研究者專注於一種對人臉識別網絡進行攻擊的新穎方法，該方法會誤導網絡將某人識別為目標人，而不是不明顯地錯誤分類，同時，因為此緣故，研究者引入了一個特定的注意力對抗攻擊生成網絡來生成假人臉圖像。為了捕獲目標人的語義信息，這項工作添加了條件變分自動編碼器和注意模塊來學習人臉之間的實例級對應關係。與傳統的雙人 GAN 不同，這項工作引入了人臉識別網絡作為第三個參與者參與生成器和判別器之間的競爭，這使得攻擊者可以更好地模仿目標人。生成的結果難以引起旁觀者註意的人臉可以逃避最先進網絡的識別，並且大多數人都被識別為目標人。

Bibliography

```
@article{yang2021attacks,
  title={Attacks on state-of-the-art face recognition using attentional adversarial attack generative network},
  author={Yang, Lu and Song, Qing and Wu, Yingqi},
  journal={Multimedia Tools and Applications},
  volume={80},
  number={1},
  pages={855--875},
  year={2021},
  publisher={Springer}
}
```