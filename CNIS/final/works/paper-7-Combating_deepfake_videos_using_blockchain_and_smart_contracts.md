# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

Hasan HR, Salah K. Combating deepfake videos using blockchain and smart contracts. IEEE Access, 2019,7:41596−41606.

Hasan, H. R., & Salah, K. (2019). Combating deepfake videos using blockchain and smart contracts. Ieee Access, 7, 41596-41606.

Link : https://ieeexplore.ieee.org/document/8668407

Note : 尝试用区块链技术对互联网上的视频进行追踪

```
With the rise of artificial intelligence (AI) and deep learning techniques, fake digital contents have proliferated in recent years.

Fake footage, images, audios, and videos (known as deepfakes) can be a scary and dangerous phenomenon and can have the potential of altering the truth and eroding trust by giving false reality.

Proof of authenticity (PoA) of digital media is critical to help eradicate the epidemic of forged content.

Current solutions lack the ability to provide history tracking and provenance of digital media.

In this paper, we provide a solution and a general framework using Ethereum smart contracts to trace and track the provenance and history of digital content to its original source even if the digital content is copied multiple times.

The smart contract utilizes the hashes of the interplanetary file system (IPFS) used to store digital content and its metadata.

Our solution focuses on video content, but the solution framework provided in this paper is generic enough and can be applied to any other form of digital content.

Our solution relies on the principle that if the content can be credibly traced to a trusted or reputable source, the content can then be real and authentic.

The full code of the smart contract has been made publicly available at Github.
```

隨著人工智能 (AI) 和深度學習技術的興起，近年來虛假數字內容激增。假鏡頭、圖像、音頻和視頻（稱為 deepfakes）可能是一種可怕且危險的現象，並且有可能通過提供虛假現實來改變真相並削弱信任，數位媒體的真實性證明 (PoA) 對於幫助消除偽造內容的流行至關重要，而當前的解決方案缺乏提供數字媒體的歷史跟踪和出處的能力。

在該研究中，研究者們提供了一個解決方案和一個通用框架，使用以太坊智能合約來追踪和跟踪數字內容的出處和歷史到其原始來源，即使數字內容被複製多次，其智能合約利用用於存儲數字內容及其元數據的行星際文件系統 (IPFS) 的哈希值。解決方案專注於視頻內容，且此研究提供的解決方案框架足夠通用，可以應用於任何其他形式的數位內容。其解決方案依賴於這樣一個原則，即如果內容可以可靠地追溯到可信或有信譽的來源，那麼內容就可以是真實的和真實的。

Bibliography

```
@article{hasan2019combating,
  title={Combating deepfake videos using blockchain and smart contracts},
  author={Hasan, Haya R and Salah, Khaled},
  journal={Ieee Access},
  volume={7},
  pages={41596--41606},
  year={2019},
  publisher={IEEE}
}
```

Fig. 1: System overview highlighting key components of proposed solution

Fig. 2: Tracing video source origin using the proposed solution

圖 1：系統概述突出了建議解決方案的關鍵組件

圖 2：使用建議的解決方案跟踪視頻源來源

## I. INTRODUCTION

The recent rise of AI, deep learning, and image processing have led the way to the production of deepfake videos. 

A video as short as one minute of the former U.S. President Barack Obama went viral in April 2018, in which
Obama was seen to say things he never said. 

Deepfake videos are dangerous, and can have the potential to undermine truth, confuse viewers and accurately fake reality. 

With the advent of social networks, proliferation of such content can be unstoppable and can potentially exacerbate problems related to misinformation and conspiracy theories.

In some early examples of deepfakes, a large number of famous political leaders, actresses, comedians, and entertainers had their faces stolen and weaved into porn videos.


Deepfake videos are far more realistic and much easier to make than traditional Hollywood-like fake videos which are typically done manually using image manipulation tools like Adobe Photoshop. 

Deepfake videos make use of deep learning techniques with input of large samples of video images to achieve face swapping. 

The higher the number of samples, the more realistic the outcome becomes. 

The Obama video was fed with more than 56 hours of sample recordings in order to make it extremely real and believable. 

Deepfake videos were not a huge concern at the beginning when they first appeared targeting celebrities.

The authors in D. Stover et al., L. Floridi et al. describe deepfake videos as a data catastrophe and called for incentivizing the general public to make good use of new technologies and to post ethically and responsibly digital content on social media outlets.

It is crucial to have techniques to detect, fight, and combat deepfake digital content that may include fake videos, images, paintings, audios, and so on.

Achieving this purpose is not difficult if there is a credible, secure, and trusted way to trace the history of digital content. 

Users should be given access to a trusted data provenance of the digital content, and be able to track back an item in history to prove its originality and authenticity. 

This mechanism can help assist users from being tricked or lured into believing in fake digital content.

Current solutions are available to prove the authenticity of physical (and not digital) artwork.

For instance, a certificate of authenticity (COA) is given with the purchase of an artwork.

Moreover, it is possible to forge this certificate or to find it unsigned from a known and trusted authority. 

Moreover, artwork bought from a secondary market is much harder to prove its origin.

The only approach currently sought, is by manually asking the gallery or the product source for the COA they have from the previous owners as well as their receipts. 

In a way, the buyer is left with substantial manual work and checking to do to achieve accurate artwork provenance.

As of today, there are no established methods for checking the originality of an online posted or published digital video, audio, or image. 

The idea to subject such digital content to a COA is not feasible. 

It is extremely difficult to determine in a credible and trusted way the true origin of a posted digital item. 

A typical user usually uses online search engines to try to find relevant posts, blogs or reviews on the digital media to judge its authenticity. 

Hence, there is an immense need for a Proof of Authenticity (PoA) system for online digital content to identify trusted published sources and therefore be able to combat deepfake videos, audios, and images.

最近人工智能、深度學習和圖像處理的興起為深度偽造視頻的製作開闢了道路。2018 年 4 月，一段短短一分鐘的美國前總統奧巴馬的視頻在網上瘋傳。
有人看到奧巴馬說了他從未說過的話。Deepfake 視頻很危險，可能會破壞真相、迷惑觀眾並準確地偽造現實。

隨著社交網絡的出現，此類內容的擴散勢不可擋，並可能加劇與錯誤信息和陰謀論相關的問題。在一些深度偽造的早期例子中，大量著名的政治領袖、女演員、喜劇演員和藝人的臉被盜並被編入色情視頻。


Deepfake 視頻比傳統的好萊塢式假視頻更逼真，更容易製作，後者通常使用 Adobe Photoshop 等圖像處理工具手動完成。Deepfake 視頻利用深度學習技術，輸入大量視頻圖像樣本來實現人臉交換而且樣本數量越多，結果就越真實。奧巴馬的視頻提供了超過 56 小時的樣本錄音，以使其極其真實和可信。

當 Deepfake 視頻首次出現針對名人時，它們一開始並不是一個大問題，D. Stover 等人、L. Floridi 等人，將 Deepfake 視頻描述為數據災難，並呼籲激勵公眾充分利用新技術，並在社交媒體上發布符合道德和負責任的數字內容。

擁有檢測、打擊和打擊可能包括假視頻、圖像、繪畫、音頻等的 deepfake 數字內容的技術至關重要。如果有一種可靠、安全和值得信賴的方式來追溯數字內容的歷史，那麼實現這一目標並不難，其用戶應有權訪問數字內容的可信數據來源，並能夠追溯歷史項目以證明其原創性和真實性，而這種機制可以幫助用戶避免被欺騙或誘騙相信虛假的數字內容。當前的解決方案可用於證明物理（而非數字）藝術品的真實性。以購買藝術品時會獲得一份真品證書 (COA) 來說，有可能偽造此證書或發現它未從已知且受信任的機構簽名的可能，同時從二級市場購買的藝術品更難證明其來源。

目前尋求的唯一方法是手動向畫廊或產品來源詢問他們從以前的所有者那裡獲得的 COA 以及他們的收據。在某種程度上，買家需要做大量的手工工作和檢查才能獲得準確的藝術品出處。到目前為止，還沒有確定的方法來檢查在線發布或發布的數字視頻、音頻或圖像的原創性。將此類數字內容納入 COA 的想法是不可行的。以可信和可信的方式確定發布的數字項目的真實來源是極其困難的，而一個典型的用戶通常使用在線搜索引擎試圖在數字媒體上找到相關的帖子、博客或評論，以判斷其真實性，也因此非常需要用於在線數字內容的真實性證明 (PoA) 系統，以識別可信的發布來源，從而能夠打擊深度偽造的視頻、音頻和圖像。


In this paper we present a decentralized Proof of Authenticity (PoA) system using the disruptive technology blockchain.

Blockchain has the ability to provide immutable and tamperproof data and transactions in a decentralized distributed ledger. 

Blockchain applicability is immense, and the technology poised to transform and impact across many businesses, industries, and domains as those in finance, the food industry, supply chain management, health management, IoT, to name just a few.

Blockchain has capabilities to provide key features that can be utilized for proving authenticity and originality of digital assets in a way that is decentralized, highly trusted and secure.

S. Singh et al., K. Biswas et al., with tamper-proof records, logs, and transactions which are openly accessible to all in case of permissionless blockchain, or restricted to certain participants in case of permissioned blockchain. 

For deepfakes, the permissionless or public blockchain is the most suitable. 

We base our solution in this paper on the public Ethereum blockchain with smart contracts to govern and capture the history of transactions made to digital content.

In this paper, we propose a blockchain-based solution and a generic framework for the proof of authenticity of digital assets that may include videos, audios, images, etc.

Our solution allows for publicly accessible, trusted, and credible data provenance, with tracking and tracing history of a published online video. 

Our solution focuses on video content, but the solution framework provided in this paper is generic enough and can be applied to any other form of digital content as audios and images. 

該研究提出了一個使用顛覆性技術區塊鏈的去中心化真實性證明 (PoA) 系統，其區塊鏈有能力在去中心化的分佈式賬本中提供不可變和防篡改的數據和交易。，同時區塊鏈的適用性是巨大的，該技術有望在金融、食品行業、供應鏈管理、健康管理、物聯網等許多企業、行業和領域發生變革和影響。而且區塊鏈具有提供關鍵功能的能力，可用於以分散、高度可信和安全的方式證明數字資產的真實性和原創性。

S. Singh 等人，K. Biswas 等人，具有防篡改記錄、日誌和交易，在無許可區塊鏈的情況下對所有人公開訪問，或者在許可區塊鏈的情況下僅限於某些參與者。對於 Deepfakes，無許可或公共區塊鍊是最合適的。研究者在該研究中的解決方案基於帶有智能合約的公共以太坊區塊鏈，以管理和捕獲對數字內容進行的交易歷史，同時研究者提出了一種基於區塊鏈的解決方案和一個通用框架，用於證明可能包括視頻、音頻、圖像等的數字資產的真實性，其解決方案允許公開訪問、可信和可信的數據來源，並跟踪和追溯已發布的在線視頻的歷史記錄。而且此解決方案專注於視頻內容，但本文提供的解決方案框架足夠通用，可以應用於任何其他形式的數字內容，如音頻和圖像。

The primary contributions of our paper can be summarized as follows:
- We present an Ethereum blockchain-based solution that establishes authenticity of digital content by providing credible and secure traceability to a trusted artist or publishing source. 

Throughout the paper, the term ”artist” is referred to the creator or publisher of the digital content.

Artists can include freelance or employed photographers, paparazzi, journalists, reporters, etc.

- We present the system architecture and design details with entity relations, sequence diagrams, and algorithms
used for Ethereum smart contracts to control and govern interactions and transactions among participants.

- We integrate into our blockchain-based system design key features of the InterPlanetary File System (IPFS) decentralized storage and reputation system, Ethereum Name service, as well as other off-chain resources to access an artist’s profile.

- We present the full implementation smart contract code1 as well as testing details.

- We provide testing details to show the correct system functionality. 

We also provide a discussion on cost estimation and security analysis of our solution.

The remainder of this paper is organized as follows.

Section II provides the related work. 

Section III presents the proposed blockchain solution.

Section IV describes the implementation and testing details. 

Section V discusses the cost and security analysis of the implemented solution and Section VI concludes the paper.

## VI. CONCLUSION

In this paper, we have presented a blockchain-based solution for proof of authenticity of digital videos in which a secure and trusted traceability to the original video creator or source can be established, in a decentralized manner. 

Our solution makes use of a decentralized storage system IPFS, Ethereum name service, and decentralized reputation system. 

Our proposed solution framework, system design, algorithms, sequence diagrams, and implementation and testing details are generic enough and can be applied to other types of digital content such as audios, photos, images, and manuscripts. 

Our solution can help combat deepfake videos and audios by helping users to determine if a video or digital content is traceable to a trusted and reputable source. 

If a video or digital content is not traceable, then the digital content cannot be trusted.

Our smart contract-based solution provides a trusted way for secondary artists to request permission from the original artist to copy and edit videos. 

The full code of the smart contract has been made available at Github. 

Key features and functionality of the smart contract have been properly tested. 

We discussed how our solution meets security requirements, and is resilient against commonly known security attacks.

We estimated the operational cost in terms of Ether and Gas when deploying the smart contract on the real Ethereum network. 

The cost estimate is minimal and is always under 0.095USD per transaction.

As a future work, we are in the process of developing frontend DApps for users to automate the establishment of proof of authenticity of published videos. 

Also we plan to develop a pluggable DApp component to provide traceability and establish authenticity when playing or displaying videos within a web browser. 

Also work is underway for designing and implementing a fully functional and operational decentralized reputation system.

最後该研究提出了一種基於區塊鏈的解決方案，用於證明數字視頻的真實性，其中可以以分散的方式建立對原始視頻創建者或來源的安全和可信的可追溯性。同時解決方案利用了去中心化存儲系統 IPFS、以太坊名稱服務和去中心化信譽系統，並提出的解決方案框架、系統設計、算法、序列圖以及實現和測試細節足夠通用，可以應用於其他類型的數字內容，例如音頻、照片、圖像和手稿，另外解決方案可以幫助用戶確定視頻或數字內容是否可追溯到可信且有信譽的來源，從而幫助打擊 deepfake 視頻和音頻。

如果視頻或數字內容不可追踪，則數字內容不可信任，其基於智能合約的解決方案為二級藝術家請求原始藝術家複製和編輯視頻的許可提供了一種值得信賴的方式，而且智能合約的完整代碼已在 Github 上提供，同時智能合約的關鍵特性和功能已經過適當的測試。研究者討論了該研究的解決方案如何滿足安全要求，以及如何抵禦常見的安全攻擊。在真實的以太坊網絡上部署智能合約時，研究者估計了以太坊和天然氣的運營成本，其成本估算是最低的，每筆交易始終低於 0.095 美元。作為未來的工作，研究者正在為用戶開發前端 DApp，以自動建立已發布視頻的真實性證明。同時還計劃開發一個可插拔的 DApp 組件，以便在 Web 瀏覽器中播放或顯示視頻時提供可追溯性並建立真實性，其設計和實施功能齊全且可操作的分散信譽系統的工作也在進行中。