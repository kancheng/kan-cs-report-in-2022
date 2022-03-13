# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

下載 GPAC，理解並描述 random access 過程

在此分别使⽤了 MP4Box 和 MP4Box.js 对 MP4 档案进⾏分析


### GPAC

GPAC 是一個 LGPL v2.1 且在大多數情況下也可以在商業許可下使用的開源多媒體框架，其專案提供了使用者在處理、檢查、打包、流式傳輸、播放和與媒體內容交互的工具。此類內容可以為音頻、影像、字幕、元數據、可縮放圖形、加密媒體、2D/3D 圖形和 ECMAScript 等任意組合。GPAC 以其廣泛的 MP4/ISOBMFF 功能而聞名，深受影像愛好者、學術研究人員、標準化機構和專業廣播公司的歡迎。

官方文件 : https://doxygen.gpac.io/


```
mp4box -info test.mp4
```

```
(base) PS D:\gpac> mp4box -info test.mp4
[isom/avcc] Missing REXT profile signaling, patching.
# Movie Info - 2 tracks - TimeScale 1000
Duration 00:03:21.550
Fragmented: no
Major Brand isom - version 512 - compatible brands: isom iso2 avc1 mp41
Created: UNKNOWN DATE

iTunes Info:
        title: Xi ?/Main Theme (Orchestral Cover/Recreation) ??Mobile Suit Gundam Hathaway's Flash OST ???柴??萸?扼 
        artist: itsthewhiteowl
        created: 20210427
        tool: Lavf58.45.100
        comment: It seems like I can't get enough of doing Hathaway music, so here is the last one before the movie and soundtrack come out. This is a recreation/cover of the track from the first proper trailer for the movie in March of last year, which has since been confirmed to be the main theme of the movie as well as Xi's theme, as the title implies. I always thought it was neat how the organ and synth texture complimented Sawano's usual bombastic orchestration style, but recently they played the full version of the track during one of the promotional livestreams, and I was blown away by how good it was. Sawano himself stated that Shukou Murase had a sort of Hollywood-esque direction, specifically a "Futuristic Orchestra" mix of Interstellar, Star Wars and Blade Runner. With the organ, Sawano was screaming Interstellar in this theme. Love the inspiration he took.

I tried to cover the track as accurately as I could from how much I could hear underneath all the talking in the livestream. I didn't do the soundscape-heavy intro part, though. Couldn't quite do the synth texture justice so I decided to leave it out and start right at the synth riff instead. There's also the slower arrangement, mSgH, which played in January's trailer and was likely going to be the 2nd half for ? based on how Sawano double track structures usually go. But with the original already being a little over 4 minutes long, it probably would have turned out too long. I didn't cover it here for similar reasons, and we've also heard very little of mSgH for me to call it a "recreation". But other than that, I'm quite happy with how this turned out.

Also, the most massive of thanks to my buddy J4F for handling the mixing and mastering of this track and getting it to sound better than I possibly could have with how much of a behemoth it is production-wise.

J4F:
https://www.instagram.com/jonathan_jaf/

-----------------------------------------------------------------

MY ALBUM:
https://itsthewhiteowl.bandcamp.com/releases
https://open.spotify.com/album/4lqJrbWUOuHLzwziTWcIlA?si=HeC2AtKBT4imCPv8syggjg

TWITTER:
https://twitter.com/itsthewhiteowl??

INSTAGRAM:
https://www.instagram.com/itsthewhiteowl
        sdesc: It seems like I can't get enough of doing Hathaway music, so here is the last one before the movie and soundtrack come out. This is a recreation/cover of the track from the first proper trailer for the movie in March of last year, which has since been confirmed to be the main theme of the movie as well as Xi's theme, as the title implies. I always thought it was neat how the organ and synth texture complimented Sawano's usual bombastic orchestration style, but recently they played the full version of the track during one of the promotional livestreams, and I was blown away by how good it was. Sawano himself stated that Shukou Murase had a sort of Hollywood-esque direction, specifically a "Futuristic Orchestra" mix of Interstellar, Star Wars and Blade Runner. With the organ, Sawano was screaming Interstellar in this theme. Love the inspiration he took.

I tried to cover the track as accurately as I could from how much I could hear underneath all the talking in the livestream. I didn't do the soundscape-heavy intro part, though. Couldn't quite do the synth texture justice so I decided to leave it out and start right at the synth riff instead. There's also the slower arrangement, mSgH, which played in January's trailer and was likely going to be the 2nd half for ? based on how Sawano double track structures usually go. But with the original already being a little over 4 minutes long, it probably would have turned out too long. I didn't cover it here for similar reasons, and we've also heard very little of mSgH for me to call it a "recreation". But other than that, I'm quite happy with how this turned out.

Also, the most massive of thanks to my buddy J4F for handling the mixing and mastering of this track and getting it to sound better than I possibly could have with how much of a behemoth it is production-wise.

J4F:
https://www.instagram.com/jonathan_jaf/

-----------------------------------------------------------------

MY ALBUM:
https://itsthewhiteowl.bandcamp.com/releases
https://open.spotify.com/album/4lqJrbWUOuHLzwziTWcIlA?si=HeC2AtKBT4imCPv8syggjg

TWITTER:
https://twitter.com/itsthewhiteowl??

INSTAGRAM:
https://www.instagram.com/itsthewhiteowl

# Track 1 Info - ID 1 - TimeScale 15360
Media Duration 00:03:21.483  (recomputed 00:03:21.500)
Track has 1 edits: track duration is 00:03:21.484
Track flags: Enabled In Movie
Media Info: Language "Undetermined (und)" - Type "vide:avc1" - 12089 samples
Visual Sample Entry Info: width=1920 height=1080 (depth=24 bits)
Visual Track layout: x=0 y=0 width=1920 height=1080
AVC/H264 Video - Visual Size 1920 x 1080
        AVC Info: 1 SPS - 1 PPS - Profile High @ Level 4.2
        NAL Unit length bits: 32
        Pixel Aspect Ratio 1:1 - Indicated track size 1920 x 1080
        Chroma format YUV 4:2:0 - Luma bit depth 8 - chroma bit depth 8
        SPS#1 hash: B3FED58C4F26FD45CC7A1641DF0CFA83EC241ECD
        PPS#1 hash: C43D6E8706EDFBB2A06353EA8731C9724B40CD24
        RFC6381 Codec Parameters: avc1.64002A
        Average GOP length: 318 samples
        Max sample duration: 256 / 15360

# Track 2 Info - ID 2 - TimeScale 44100
Media Duration 00:03:21.549
Track has 1 edits: track duration is 00:03:21.550
Track flags: Enabled In Movie
Media Info: Language "English (eng)" - Type "soun:mp4a" - 8680 samples
Alternate Group ID 1
MPEG-4 Audio AAC LC (AOT=2 implicit) - 2 Channel(s) - SampleRate 44100
        RFC6381 Codec Parameters: mp4a.40.2
        All samples are sync
        Max sample duration: 1024 / 44100

(base) PS D:\gpac>
```

可以看到結果有两个 Track，第⼀個是影像，第⼆個是⾳頻與其它資訊。

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


### Random Access 過程與說明

尋找專案中有關 Random Access 的部分，利用 find 和 grep 指令。

```
find . -name "*.*" |xargs grep "random access" *.*
```

```
./applications/mp4box/fileimport.c
./applications/mp4box/main.c
./extra_lib/include/zlib/zlib.h
./include/gpac/constants.h
./include/gpac/filters.h:
./include/gpac/html5_media.h:
./include/gpac/ietf.h:
./include/gpac/internal/swf_dev.h:
./include/gpac/isomedia.h:
./include/gpac/isomedia.h:
./include/gpac/media_tools.h:
./include/gpac/mpegts.h:
./include/gpac/rtp_streamer.h:
./include/gpac/scene_manager.h: 
./share/doc/man/gpac-filters.1:mfra (bool, default: false):
./share/doc/man/mp4box.1:
./src/filters/mux_isom.c:
./src/media_tools/html5_mse.c: 
./src/media_tools/m2ts_mux.c:
./src/scene_manager/text_to_bifs.c:
```

If you want to seek a given track to a time T,
假如想要最一個文件進行隨機訪問，而該訪問進行為 T 時刻 EX : 第三分五十秒。

1. If the track contains an edit list, determine which edit contains the time T by iterating over the edits. T_movie = T_start + T’

如果軌道包含編輯列表，則通過迭代編輯確定哪個編輯包含時間 T。 T_movie = T_start + T'

先找當中有沒有 edit list，當中 edit list 實際上存著一段段的樣本信息，這裡面會有一個關鍵字段的有開始時刻(T_start)的連續樣本。這個過程就是將 T 換算成 T_start + T'，而這個 T' 就是換算出來的新的時間。

2. Convert to media time scale T_media = T_start’ + T’’

然後再將原先的過程轉換為媒體時間尺度 T_media = T_start' + T''


3. Use time-to-sample box to find the first sample prior to the given time

使用 time-to-sample box 找到給定時間之前的第一個樣本，也就是對應的樣本編號。

4. Consult the sync sample table to seek to which sample is closest to, but prior to, the sample found above

因為找到的樣本不一定是可以解的，為了保險會先查閱同步樣本表以尋找最接近但先於上面所找到的樣本的樣本編號。

5. Use the sample-to-chunk table to determine in which chunk this sample is located.

使用 sample-to-chunk 來確定該樣本位於哪個 chunk 中。

6. use the chunk offset box to figure out where that chunk begins

找到後根據使用 the chunk offset box 來確定該 chunk 開始的物理存儲位置。

7. Starting from this offset, you can use the information contained in the sample-to-chunk box and the sample size box to figure out where within this chunk the sample in question is located.

從這個偏移量開始，您可以使用包含在 sample-to-chunk box 和 the sample size box 中的信息來確定有問題的樣本在這個 chunk 中的位置。

### Random Access 技術文件

本節描述如何尋找。

查找主要通過使⽤ sample table box 中的⼦ box 來完成。如果 edit list 存在，也必須查閱它。 如果你想尋找⼀個給定的軌道到⼀個時間 T，其中 T 是movie header box 的 time scale，執⾏以下操作:


1. If the track contains an edit list, determine which edit contains the time T by iterating over the edits. The start time of the edit in the movie time scale must then be subtracted from the time T to generate T', the duration into the edit in the movie time scale. T' is next converted to the time scale of the track's media to generate T''. Finally, the time in the media scale to use is calculated by adding the media start time of the edit to T''.

（如果 track 有 edit list，遍歷所有 edit，找到T在哪⼀個 track ⾥。將 edit 的開始時間轉換為 movie 的 time scale 為單位得到 edit_T，T 減去 edit_T，得到 T'，也就是在 edit ⾥⾯的持續時間。將 T' 轉換成 track 媒體 的 time scale，得到 T''。最後將 T'' 加上 edit_T，可以得到以 track 媒體 的 time scale 為單位的 T'''，⽽這個 T''' 就是後續⽤來求 sample 的時間）

2. The time-to-sample box for a track indicates what times are associated with which sample for that track. Use this box to find the first sample prior to the given time.

3. The sample that was located in step 1 may not be a random access point. Locating the nearest random access point requires consulting two boxes. The sync sample table indicates which samples are in fact random access points. Using this table, you can locate which is the first sync sample prior to the specified time. The absence of the sync sample table indicates that all samples are synchronization points, and makes this problem easy. The shadow sync box gives the opportunity for a content author to provide samples that are not delivered in the normal course of delivery, but which can be inserted to provide additional random access points. This improves random access without impacting bitrate during normal delivery. This box maps samples that are not random access points to alternate samples that are. You should also consult this table if present to find the first shadow sync sample prior to the sample in question. Having consulted the sync sample table and the shadow sync table, you probably wish to seek to whichever resultant sample is closest to, but prior to, the sample found in step 1.

4. At this point you know the sample that will be used for random access. Use the sample-to-chunk table to determine in which chunk this sample is located.

5. Knowing which chunk contained the sample in question, use the chunk offset box to figure out where that chunk begins.

6. Starting from this offset, you can use the information contained in the sample-to-chunk box and the sample size box to figure out where within this chunk the sample in question is located. This is the desired information.


### MP4 分析

https://github.com/gpac/mp4box.js

https://gpac.github.io/mp4box.js/test/filereader.html

https://gpac.github.io/mp4box.js/


### Code 分析

```
/ ... /gpac-master/src/isomedia/isom_read.c
/ ... /gpac-master/src/isomedia/stbl_read.c
```


## Reference

0. https://github.com/gpac/gpac

1. https://gpac.wp.imt.fr/

2. https://its201.com/article/arau_sh/19083223

3. https://blog.csdn.net/MyArrow/article/details/39522627

4. https://www.jianshu.com/p/3214345d1df0

5. https://www.youtube.com/watch?v=R0STwCVDizE

6. http://www.4k8k.xyz/article/u013354805/51547469
