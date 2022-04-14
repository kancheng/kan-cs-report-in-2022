# Note - CNIS 密码编码学与网络信息安全 Cryptography and Network Information Security

## 1. 碰撞

理論上來說，當類似於 MD5 這種摘要算法被完全攻破時，也就是說可以從摘要上恢復出任意原文。

要注意的是，其任意原文，因為所有的摘要算法的特點就是存在著一份無窮大的碰撞原文的集合。而真正的原文只是其中一份。

對應這個無窮大的集合來說，這個可能性無窮小。

## 2. SNMPV3

簡單網路管理協定（SNMP，Simple Network Management Protocol）構成了網際網路工程工作小組（IETF，Internet Engineering Task Force）定義的Internet協定族的一部分。該協定能夠支援網路管理系統，用以監測連接到網路上的裝置是否有任何引起管理上關注的情況。它由一組網路管理的標準組成，包含一個應用層協定（application layer protocol）、資料庫模式（database schema），和一組資料物件。


## Reference

1. 常用消息摘要算法简介 : https://cloud.tencent.com/developer/article/1584742

2. 常见摘要算法(如MD5, SHA1, SHA256, CRC32)的碰撞能够避免吗？ https://www.zhihu.com/question/52474834

3. 簡單網路管理協定 : https://zh.wikipedia.org/zh-tw/%E7%AE%80%E5%8D%95%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86%E5%8D%8F%E8%AE%AE