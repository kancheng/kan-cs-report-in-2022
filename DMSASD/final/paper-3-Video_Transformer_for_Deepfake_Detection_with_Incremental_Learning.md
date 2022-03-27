# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Video Transformer for Deepfake Detection with Incremental Learning

Khan, S. A., & Dai, H. (2021, October). Video Transformer for Deepfake Detection with Incremental Learning. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 1821-1828).

https://dl.acm.org/doi/abs/10.1145/3474085.3475332?sid=SCITRUS

```
Face forgery by deepfake is widely spread over the internet and this raises severe societal concerns. 

In this paper, we propose a novel video transformer with incremental learning for detecting deepfake videos. 

To better align the input face images, we use a 3D face reconstruction method to generate UV texture from a single input face image. 

The aligned face image can also provide pose, eyes blink and mouth movement information that cannot be perceived in the UV texture image, so we use both face images and their UV texture maps to extract the image features. 

We present an incremental learning strategy to fine-tune the proposed model on a smaller amount of data and achieve better deepfake detection performance. 

The comprehensive experiments on various public deepfake datasets demonstrate that the proposed video transformer model with incremental learning achieves state-of-the-art performance in the deepfake video detection task with enhanced feature learning from the sequenced data.
```

Deepfake 的面部偽造在互聯網上廣泛傳播，這引起了嚴重的社會擔憂，在該研究中，研究者提出了一種具有增量學習功能的新型影像轉換器，用於檢測深度偽造影像，而為了更好地對齊輸入人臉圖像，我們使用 3D 人臉重建方法從單個輸入人臉圖像生成 UV 紋理。其對齊的人臉圖像還可以提供在 UV 紋理圖像中無法感知的姿勢、眨眼和嘴巴運動資訊，因此研究者使用人臉圖像及其 UV 紋理圖來提取圖像特徵。最後研究者提出了一種增量學習策略，可以在更少量的資料上微調所提出的模型，並實現更好的深度偽造檢測性能。對各種公共 deepfake 數據集的綜合實驗表明，所提出的具有增量學習的視頻轉換器模型通過對序列數據的增強特徵學習，在 deepfake 視頻檢測任務中實現了最先進的性能。

Bibliography

```
@inproceedings{khan2021video,
  title={Video Transformer for Deepfake Detection with Incremental Learning},
  author={Khan, Sohail Ahmed and Dai, Hang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1821--1828},
  year={2021}
}
```

## Note
