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