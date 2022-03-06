# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

下載 GPAC，理解並描述 random access 過程

### GPAC

GPAC 是一個 LGPL v2.1 且在大多數情況下也可以在商業許可下使用的開源多媒體框架，其專案提供了使用者在處理、檢查、打包、流式傳輸、播放和與媒體內容交互的工具。此類內容可以為音頻、影像、字幕、元數據、可縮放圖形、加密媒體、2D/3D 圖形和 ECMAScript 等任意組合。GPAC 以其廣泛的 MP4/ISOBMFF 功能而聞名，深受影像愛好者、學術研究人員、標準化機構和專業廣播公司的歡迎。

官方文件 : https://doxygen.gpac.io/

```
mp4box -version
```

```
# 1
mp4box -h
# 查看 mp4box 中的所有幫助信息

# 2
mp4box -h general
# 查看 mp4box 中的通用幫助信息

# 3
mp4box -info test.mp4 
# 查看 test.mp4 文件是否有問題

# 4
mp4box -add test.mp4 test-new.mp4
# 修復 test.mp4 文件格式不標準的問題，並把新文件保存在 test-new.mp4 中

# 5
mp4box  -inter  10000 test-new.mp4 
# 解決開始播放 test-new.mp4 卡一下的問題，為 HTTP 下載快速播放有效，10000ms

# 6
mp4box -add file.avi new_file.mp4
# 把 avi 文件轉換為 mp4 文件

# 7
mp4box -hint file.mp4 
# 為 RTP 準備，此指令將為文件創建RTP提示跟踪信息。
# 這使得經典的流媒體服務器像 darwinstreamingserver 或 QuickTime 的流媒體服務器通過 RTSP／RTP 傳輸文件

# 8
mp4box -cat test1.mp4 -cat test2.mp4 -new test.mp4 
# 把 test1.mp4 和 test2.mp4 合併到一個新的文件 test.mp4 中，要求編碼參數一致

# 9
mp4box -force-cat test1.mp4 -force-cat test2.mp4 -new test.mp4 
# 把 test1.mp4 和 test2.mp4 強制合併到一個新的文件 test.mp4 中，有可能不能播放

# 10
mp4box -add video1.264 -cat video2.264 -cat video3.264 -add audio1.aac -cat audio2.aac -cat audio3.aac -new muxed.mp4 -fps 24 
# 合併多段音視頻並保持同步 

# 11
mp4box -split *time_sec* test.mp4
# 切取 test.mp4 中的前面 time_sec 秒的視頻文件

# 12
mp4box -split-size *size *test.mp4 
# 切取前面大小為 size KB的視頻文件

# 13
mp4box -split-chunk *S:E* test.mp4 
# 切取起始為 S 少，結束為 E 秒的視頻文件

# 14
mp4box -add 1.mp4#video -add 2.mp4#audio -new test.mp4
# test.mp4 由 1.mp4 中的視頻與 2.mp4 中的音頻合併生成
```


### Random Access 過程

尋找專案中有關 Random Access 的部分，利用 find 和 grep 指令。

```
find . -name "*.*" |xargs grep "random access" *.*
```

If you want to seek a given track to a time T,

- If the track contains an edit list, determine which edit contains the time T by iterating over the edits. T_movie = T_start + T’

- Convert to media time scale T_media = T_start’ + T’’

- Use time-to-sample box to find the first sample prior to the given time

- Consult the sync sample table to seek to which sample is closest to, but prior to, the sample found above

- Use the sample-to-chunk table to determine in which chunk this sample is located.

- use the chunk offset box to figure out where that chunk begins

- Starting from this offset, you can use the information contained in the sample-to-chunk box and the sample size box to figure out where within this chunk the sample in question is located.

如果你想尋找一個給定的軌道到時間 T，

- 如果軌道包含編輯列表，則通過迭代編輯確定哪個編輯包含時間 T。 T_movie = T_start + T'

- 轉換為媒體時間尺度 T_media = T_start' + T''

- 使用 time-to-sample 框找到給定時間之前的第一個樣本

- 查閱同步樣本表以尋找最接近但先於上面找到的樣本的樣本

- 使用樣本到塊表來確定該樣本位於哪個塊中。

- 使用塊偏移框來確定該塊開始的位置

- 從這個偏移量開始，您可以使用包含在樣本到塊框和样本大小框中的信息來確定有問題的樣本在這個塊中的位置。

## Reference

0. https://github.com/gpac/gpac

1. https://gpac.wp.imt.fr/

2. https://its201.com/article/arau_sh/19083223

3. https://blog.csdn.net/MyArrow/article/details/39522627

4. https://www.jianshu.com/p/3214345d1df0

5. https://www.youtube.com/watch?v=R0STwCVDizE

6. http://www.4k8k.xyz/article/u013354805/51547469
