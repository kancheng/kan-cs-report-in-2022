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


Figure 1. Adversarial attack results in our work. 

The first column is the target face. The 2nd and 4th columns are the original images and the rest are the generated images. 

Given target images, our work is to generate images similar to the original faces but classified as the target person.

图 1. 我们工作的对抗性攻击結果。第一列是目标面。 第 2 列和第 4 列是原始图像，其余的是生成的图像。给定目标图像，其工作是生成与原始人脸相似但归类为目标人物的图像。


## 1. Introduction

Neural network is widely used in different tasks in society which is profoundly changing our life. 

Good algorithm, adequate training data, and computing power make neural network supersede human in many tasks, such as face recognition. 

Face recognition can be used to determine which one the face images belong to or whether the two face images belong to the same one. 

Applications based on this technology are gradually adopted in some important tasks, such as identity authentication in a railway station and for payment. 

Unfortunately, it has been shown that face recognition network can be deceived inconspicuously by mildly changing inputs maliciously. 

The changed inputs are named adversarial examples which implement adversarial attacks on networks.

Szegedy et al. present that adversarial attacks can be implemented by applying an imperceptible perturbation which is hard to be observed for human eyes for the first time. 

Following the work of Szegedy, many works focus on how to craft adversarial examples to attack neural networks.

Neural network is gradually under suspicion. 

The works on adversarial attacks can promote the development of neural network.

Akhtar et al. review these works’ contributions in the real-world scenarios. Illuminated by predecessor’s works, we also do some research about adversarial attack.

Most of adversarial attacks aim at misleading classifier to a false label, not a determined specific label. 

Besides, attacks on image classifier can not be against face recognition networks. 

Existing works produce perturbation on the images, do some makeup to faces and add eyeglass, hat or occlusions to faces.

And their adversarial examples are fixed by the algorithms which are not flexible for attacks. 

These algorithms can not accept any images as inputs. 

Our goal is to generate face images which are similar to the original images but can be classified as the target person shown in Fig. 1. 

The method manipulating the intensity of input images directly is intensity-based.

Our work uses geometry-based method to generate adversarial examples. 

In our work, we use generative adversarial net (GAN) to produce adversarial examples which are not limited by data, algorithms or target networks. 


It can accept any faces as inputs and convert them to adversarial examples for attacks. 

To generate adversarial examples, we present A3GN to produce the fake image whose appearance is similar to the origin but is able to be classified as the target person.

In face verification domain, whether the two faces belong to one person is based on the cosine distance between feature map in the last layer not based on the probability for each category. 

So $A_{3}GN$ pays more attention to the exploration of feature distribution for faces. 

To get the instance information, we introduce a conditional variational autoencoder to get the latent code from the target face, and meanwhile, attentional modules are provided to capture more feature representation and facial dependencies of the target face. 

For adversarial examples, $A_{3}GN$ adopts two discriminators – one for estimating whether the generated faces are real called normal discriminator, another for estimating whether the generated faces can be classified as the target person called instance discriminator.

Meanwhile, cosine loss is introduced to promise that the fake images can be classified as the target person by the target model. 

Our main contributions can be summarized into three-fold:

- We focus on a novel way of attacking against state-ofthe-art face recognition networks.

They will be misled to identify someone as the target person not misclassify inconspicuously in face verification according to the feature map not the probability.

- GAN is introduced to generate the adversarial examples different from traditional intensity-based attacks. 

Meanwhile, this work presents a new GAN named $A_{3}GN$ to generate adversarial examples which are similar to the origins but have the same feature representation as the target face.

- Good performance of $A_{3}GN$ can be shown by a set of evaluation criteria in physical likeness, similarity score, and accuracy of recognition.


神經網絡廣泛應用於社會中的不同任務，深刻地改變著我們的生活。其良好的算法、充足的訓練數據和計算能力使神經網絡在許多任務中取代了人類，例如人臉識別，而人臉識別可用於確定人臉圖像屬於哪一張或兩張人臉圖像是否屬於同一張。基於該技術的應用逐漸被應用在一些如火車站的身份認證和支付的重要任務。不幸的是識別網絡可以通過惡意溫和地改變輸入而被不引人注意地欺騙，更改的輸入被命名為對抗性示例，它們對網絡實施對抗性攻擊。

Szegedy et al 等人發現，目前對抗性攻擊可以通過應用難以察覺的擾動來實現，這是人眼第一次難以觀察到的擾動，而在 Szegedy 的工作之後，許多工作都集中在如何製作對抗性示例來攻擊神經網絡，使神經網絡逐漸受到質疑。對抗性攻擊的工作可以促進神經網絡的發展，Akhtar 等人則回顧這些作品在現實世界場景中的貢獻。在前人作品的啟發下，該研究還對對抗性攻擊進行了一些研究。大多數對抗性攻擊旨在將分類器誤導為錯誤標籤，而不是確定的特定標籤。

此外對圖像分類器的攻擊不能針對人臉識別網絡，現有作品會對圖像產生擾動，對面部進行一些化妝，並在面部添加眼鏡、帽子或遮擋物。其對抗性例子是由不靈活的攻擊算法固定的，而這些算法不能接受任何圖像作為輸入。該研究者的目標是生成與原始圖像相似但可以分類為目標人的人臉圖像，如圖中所示，而直接操縱輸入圖像強度的方法是基於強度的。另外研究者的工作使用基於幾何的方法來生成對抗樣本。在該研究的工作中，研究者們使用生成對抗網絡 (GAN) 來生成不受數據、算法或目標網絡限制的對抗樣本。


它可以接受任何人臉作為輸入，並將其轉換為對抗樣本以進行攻擊。為了生成對抗樣本，該工作提出了 $A_{3}GN$ 來生成外觀與原點相似但能夠被歸類為目標人的假圖像。在人臉驗證領域，兩張人臉是否屬於一個人是基於最後一層特徵圖之間的餘弦距離，而不是基於每個類別的概率。所以 $A_{3}GN$ 更加關注人臉特徵分佈的探索。為了獲取實例信息，該研究引入了條件變分自動編碼器來從目標人臉中獲取潛在代碼，同時提供了注意力模塊來捕獲目標人臉的更多特徵表示和臉部依賴關係。對於對抗樣本，$A_{3}GN$ 採用了兩種判別器——一個用於估計生成的人臉是否真實，稱為正常判別器，另一個用於估計生成的人臉是否可以分類為目標人，稱為實例判別器。

同時，引入餘弦損失以保證假圖像可以被目標模型分類為目標人物。

研究者的主要貢獻可以概括為三方面：

- 研究者專注於攻擊最先進的人臉識別網絡的新方法。

他們會被誤導，將某人識別為目標人，而不是根據特徵圖而不是概率在人臉驗證中不明顯地錯誤分類。

- 引入 GAN 以生成與傳統基於強度的攻擊不同的對抗性示例。

同時，這項工作提出了一種名為 $A_{3}GN$ 的新 GAN，用於生成與原點相似但與目標人臉具有相同特徵表示的對抗樣本。

- $A_{3}GN$ 的良好表現可以通過一組評估標准在身體相似度、相似度得分和識別準確度方面表現出來。