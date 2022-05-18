# CNIS - 密码编码学与网络信息安全 Cryptography and Network Information Security

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## 個人 2021

```
1.
在深圳研究院的 WLAN 中，从请求接入开始，分别运行一种 http 应用以及一种 https 应用，使用 Wireshark 网络抓包工具，进行抓包，并对抓取的信息流程进行分析，试论 Wi-Fi 探针的原理与应用。


2. 
A) 试论量子计算机的发展对信息安全领域的影响，包括带来的挑战与对策。
B) 综述可信计算技术的最新发展，试论如何利用可信，计算技术提升目前的系统安全。
```

## 團隊 2021

```
A) 利用 OpenSSL 或其他软件产生 X509 PKI 证书，并针对一种应用场景实现通信双方的相互身份鉴别，并用 D-H 产生对称密钥，进行信息加密保护。

B) 利用信息安全技术，提升一种智能应用的安全性。并试论智能安全的实践意义。
```

## 個人 2022

```
1. 在深圳研究院的 WLAN 中，从请求接入开始，分别运行一种 http 应用以及一种 https 应用，使用 Wireshark 网络抓包工具，进行抓包，并对抓取的信息流程进行分析，试论 Wi-Fi 探针的原理与应用，以及在軌跡跟蹤的應用。

2. 試論聯邦學習的安全特點以及隱私計算的關係，包括應用現狀及發展機遇。
```

## 團隊 2022

```
基於 PKI 架構，設計一個安全的即時通訊方法，實現團隊成員文件安全共享及接收功能。
```


## Details

### PHP

1. XAMPP

2. PHP

```
php -h
```

```
(base) PS D:\git-project\github\kan-cs-report-in-2022\CNIS\final\code> php -h
Usage: php [options] [-f] <file> [--] [args...]
   php [options] -r <code> [--] [args...]
   php [options] [-B <begin_code>] -R <code> [-E <end_code>] [--] [args...]
   php [options] [-B <begin_code>] -F <file> [-E <end_code>] [--] [args...]
   php [options] -S <addr>:<port> [-t docroot] [router]
   php [options] -- [args...]
   php [options] -a

  -a               Run as interactive shell
  -c <path>|<file> Look for php.ini file in this directory
  -n               No configuration (ini) files will be used
  -d foo[=bar]     Define INI entry foo with value 'bar'
  -e               Generate extended information for debugger/profiler
  -f <file>        Parse and execute <file>.
  -h               This help
  -i               PHP information
  -l               Syntax check only (lint)
  -m               Show compiled in modules
  -r <code>        Run PHP <code> without using script tags <?..?>
  -B <begin_code>  Run PHP <begin_code> before processing input lines
  -R <code>        Run PHP <code> for every input line
  -F <file>        Parse and execute <file> for every input line
  -E <end_code>    Run PHP <end_code> after processing all input lines
  -H               Hide any passed arguments from external tools.
  -S <addr>:<port> Run with built-in web server.
  -t <docroot>     Specify document root <docroot> for built-in web server.
  -s               Output HTML syntax highlighted source.
  -v               Version number
  -w               Output source with stripped comments and whitespace.
  -z <file>        Load Zend extension <file>.

  args...          Arguments passed to script. Use -- args when first argument
                   starts with - or script is read from stdin

  --ini            Show configuration file names

  --rf <name>      Show information about function <name>.
  --rc <name>      Show information about class <name>.
  --re <name>      Show information about extension <name>.
  --rz <name>      Show information about Zend extension <name>.
  --ri <name>      Show configuration for extension <name>.

(base) PS D:\git-project\github\kan-cs-report-in-2022\CNIS\final\code>
```

```
php -S 127.0.0.1:8989
```


## Reference

1. PKI 和 X509 证书 : https://blog.csdn.net/code_segment/article/details/89647358

2. 后量子时代的数据保护 : https://www.pwccn.com/zh/issues/cybersecurity-and-data-privacy/digital-security-post-quantum-world-mar2021.pdf

3. 朱晓波、陆朝阳、潘建伟：量子计算 - 后摩尔时代计算能力提升的解决方案 : https://www.cas.cn/zjs/202203/t20220302_4826718.shtml

4. 量子信息技术发展与应用研究报告 : https://pdf.dfcfw.com/pdf/H3_AP202112291537269645_1.pdf?1640773504000.pdf

5. How to create your own PKI with openssl : https://evilshit.wordpress.com/2013/06/19/how-to-create-your-own-pki-with-openssl/

6. Tutorial: Using OpenSSL to create test certificates : https://docs.microsoft.com/en-us/azure/iot-hub/tutorial-x509-openssl

7. How to Use OpenSSL with a Windows Certificate Authority to Generate TLS Certificates to use with XenServer : https://support.citrix.com/article/CTX128656

8. 量子计算综述报告 : https://www.163.com/dy/article/GP6O5B960552NPC3.html

9. Research Directions in Quantum Cryptography : https://www.researchgate.net/publication/220840552_Research_Directions_in_Quantum_Cryptography

10. State-of-the-Art Survey of Quantum Cryptography : https://link.springer.com/article/10.1007/s11831-021-09561-2

11. Quantum cryptography: A survey : https://dl.acm.org/doi/10.1145/1242471.1242474

12. 量子技术时代下的信息安全 : https://www.jsjkx.com/CN/article/openArticlePDF.jsp?id=688

13. 信息安全 : https://zh.wikipedia.org/wiki/%E4%BF%A1%E6%81%AF%E5%AE%89%E5%85%A8

14. 網站常見的資安問題 : https://medium.com/schaoss-blog/%E5%89%8D%E7%AB%AF%E4%B8%89%E5%8D%81-29-web-%E7%B6%B2%E7%AB%99%E5%B8%B8%E8%A6%8B%E7%9A%84%E8%B3%87%E5%AE%89%E5%95%8F%E9%A1%8C%E6%9C%89%E5%93%AA%E4%BA%9B-bc47b572d94d

15. Fundamentals of Top 10 Open Web Application Security Project : https://medium.com/@yalexcortes/fundamentals-of-top-10-open-web-application-security-project-af8d5b7aa7dd

16. 机器学习系统的隐私和安全问题综述 : https://crad.ict.ac.cn/CN/10.7544/issn1000-1239.2019.20190437

17. 人工智能系统安全与隐私风险 - Security and Privacy Risks in Artificial Intelligence Systems : https://scholars.cityu.edu.hk/en/publications/untitled(03889802-b8e3-4be4-8a82-59b4c2a5bc76).html

18. Detecting web attacks with end-to-end deep learning : https://jisajournal.springeropen.com/articles/10.1186/s13174-019-0115-x

19. 量子世代下的密碼學：機會與挑戰 : https://www.iis.sinica.edu.tw/zh/page/report/8106.html

20. 量子密碼學 : https://zh.wikipedia.org/wiki/%E9%87%8F%E5%AD%90%E5%AF%86%E7%A2%BC%E5%AD%B8

21. 可信计算/可信用计算（Trusted Computing，TC）: https://zh.wikipedia.org/wiki/%E5%8F%AF%E4%BF%A1%E8%AE%A1%E7%AE%97

22. 量子计算与量子密码的原理及研究进展综述, Principle and Research Progress of Quantum Computation and Quantum Cryptography : https://crad.ict.ac.cn/CN/10.7544/issn1000-1239.2020.20200615

23. 量子密码学综述 : https://blog.csdn.net/weixin_37773108/article/details/106064311

24. 可信计算概述 : https://zhuanlan.zhihu.com/p/80413237

25. Wireshark 抓包使用指南 : https://zhuanlan.zhihu.com/p/82498482

26. HTTPS 运行流程 : https://zhuanlan.zhihu.com/p/60033345

27. NodeJS-Web Server I : https://ithelp.ithome.com.tw/articles/10273478

28. Apache 安裝及設定 PHP 環境、SSL 及查看 DNS: https://hoohoo.top/blog/apache-installs-and-sets-php-environments-ssl-and-view-dns/

29. SSL 憑證加密網站, 從 HTTP 到 HTTPS : https://www.j2h.tw/bbs/bbs16/806.html

30. XAMPP 設定本地端 (localhost) SSL(https) 方法 10 步驟 : https://www.barryblogs.com/xampp-localhost-ssl-certificate/

31. 關於 PHP 設定 HTTPS 的問題 : https://tw511.com/a/01/17375.html

32. PHP 实现 http 与 https 转化 : https://blog.csdn.net/jimlong/article/details/50549712

33. PHP - 利用 Openssl 實作 ssl 網頁加密 : https://joe01032002.pixnet.net/blog/post/92665237

34. Create a self signed certificate in Windows [Full Guide]: https://windowsreport.com/create-self-signed-certificate/

35. 如何刪除裝置憑證？ (Windows 10): https://support.hdeone.com/hc/zh-tw/articles/360014871753-%E5%A6%82%E4%BD%95%E5%88%AA%E9%99%A4%E8%A3%9D%E7%BD%AE%E6%86%91%E8%AD%89-Windows-10-

36. Wireshark 抓包分析 HTTP 和 HTTPS : https://bjjdkp.github.io/post/wireshark-http-https/

37. wireshark 抓包比较 http 与 https 头信息 : https://blog.csdn.net/dengjili/article/details/88745875

38. wireshark 如何扑捉无线局域网数据？ : https://www.zhihu.com/question/28838507/answer/424537660

39. Research on trusted computing and its development : https://www.researchgate.net/publication/220361980_Research_on_trusted_computing_and_its_development

40. WiFi 探针的原理与安全 : https://www.yisu.com/zixun/76931.html

41. WiFi 探针技术 : https://zhuanlan.zhihu.com/p/98103330

42. WiFi 探针获取无线网络信息技术简介与测试 : https://www.anquanke.com/post/id/181171

43. 量子计算与机器学习: 量子生成对抗网络 QGAN : https://zhuanlan.zhihu.com/p/72534334

44. TPM分析笔记 TPM 2.0 规范文档 : https://blog.csdn.net/xy010902100449/article/details/123312545

45. OpticalFlow-in-Deepfake-Detection-Application : https://github.com/NUISTGY/OpticalFlow-in-Deepfake-Detection-Application

46. Deepfakes-Detection-Papers : https://github.com/chenshen03/Deepfakes-Detection-Papers

47. 微軟突破技術限制，實現拓撲量子位元 : https://www.ithome.com.tw/news/149916

48. Federated Learning 聯邦學習簡介 : https://biic.ee.nthu.edu.tw/blog-detail.php?id=2

49. linux tcpdump 命令以及结果分析 : https://blog.csdn.net/redsuntim/article/details/8892339

50. Linux tcpdump 命令 : https://www.runoob.com/linux/linux-comm-tcpdump.html

51. Digital footprints: Using WiFi probe and locational data to analyze human mobility trajectories in cities : https://www.sciencedirect.com/science/article/pii/S0198971517305914

52. tcpdump 抓包分析 : https://blog.csdn.net/daidadeguaiguai/article/details/119758391
