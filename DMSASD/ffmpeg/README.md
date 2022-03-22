# DMSASD - 数字媒体软件与系统开发 - Digital Media Software And System Development

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業

## Details

FFMGEG 下載，並說明 output_example 。

## Code

1. https://github.com/FFmpeg/FFmpeg

2. https://www.ffmpeg.org/download.html

3. https://ffmpeg.org/doxygen/0.6/output-example_8c.html

4. https://ffmpeg.org/doxygen/0.6/output-example_8c-source.html

5. https://libav.org/documentation/doxygen/master/output_8c-example.html

6. https://ffmpeg.org/doxygen/trunk/output-example_8c.html

7. https://ffmpeg.org/doxygen/trunk/output-example_8c-source.html

8. https://ffmpeg.org/doxygen/trunk/avformat_8h-source.html

9. https://ffmpeg.org/doxygen/trunk/swscale_8h-source.html


## GitHub 版本追蹤

將所有 Log 紀錄用指令輸出。

```
git log > log.txt
```
0.6 還有，但在 0.7 版此檔案就已經被拔掉，其大的版本可以在 v0.6.1 中找到。

從 GitHub 的版本號中可以看到是由 Michael Niedermayer 所提交的合併更動時消失。

```
commit fbe02459dc4f3c8f4d758c1a90ed8e35a800f3b9
Merge: 9a1963fbb8 b4675d0fbf
Author: Michael Niedermayer <michael@niedermayer.cc>
Date:   Mon Jul 16 01:32:52 2012 +0200

    Merge remote-tracking branch 'qatar/master'
    
    * qatar/master:
      configure: Check for CommandLineToArgvW
      vc1dec: Do not use random pred_flag if motion vector data is skipped
      vp8: Enclose pthread function calls in ifdefs
      snow: refactor code to work around a compiler bug in MSVC.
      vp8: Include the thread headers before using the pthread types
      configure: Check for getaddrinfo in ws2tcpip.h, too
      vp8: implement sliced threading
      vp8: move data from VP8Context->VP8Macroblock
      vp8: refactor decoding a single mb_row
      doc: update api changes with the right commit hashes
      mem: introduce av_malloc_array and av_mallocz_array
    
    Conflicts:
            configure
            doc/APIchanges
            libavcodec/vp8.c
            libavutil/mem.h
            libavutil/version.h
    
    Merged-by: Michael Niedermayer <michaelni@gmx.at>
```

追檔案更動

```
git log --full-history -- libavformat/output-example.c
```

最後發現被搬移至此，更後面就沒有該範例的存在。

```
 libavformat/output-example.c → doc/examples/output.c
```

Log

```
commit ab81f24ad43bddf77ddd25cba86780c1c884996c
Author: Diego Biurrun <diego@biurrun.de>
Date:   Sat Nov 2 17:05:28 2013 +0100

    build: Integrate multilibrary examples into the build system

    This includes moving libavformat/output-example to doc/examples/output.
```
https://github.com/FFmpeg/FFmpeg/commit/ab81f24ad43bddf77ddd25cba86780c1c884996c


追檔案更動

```
git log --full-history -- doc/examples/output.c
```

```
git reset --hard HEAD^
git reset --hard [COMMIT]
```


## Reference

0. https://blog.csdn.net/fireroll/article/details/49814903

1. https://blog.csdn.net/fireroll/article/details/24141151

2. https://cxybb.com/article/dj0379/7824404

3. https://github.com/xufuji456/FFmpegAndroid

4. https://www.gushiciku.cn/pl/gW3Z/zh-tw

5. https://blog.csdn.net/qq_33750826/article/details/107524418

6. https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/694897/

7. https://blog.csdn.net/teng_ontheway/article/details/50217953

8. https://batchloaf.wordpress.com/2017/02/12/a-simple-way-to-read-and-write-audio-and-video-files-in-c-using-ffmpeg-part-2-video/

## Logs

1. ./configure --disable-x86asm

```
(base) HaoyeMacBookPro:ffmpeg kancheng$ ./configure --disable-x86asm
install prefix            /usr/local
source path               .
C compiler                gcc
C library                 
ARCH                      x86 (generic)
big-endian                no
runtime cpu detection     yes
standalone assembly       no
x86 assembler             nasm
MMX enabled               yes
MMXEXT enabled            yes
3DNow! enabled            yes
3DNow! extended enabled   yes
SSE enabled               yes
SSSE3 enabled             yes
AESNI enabled             yes
AVX enabled               yes
AVX2 enabled              yes
AVX-512 enabled           yes
AVX-512ICL enabled        yes
XOP enabled               yes
FMA3 enabled              yes
FMA4 enabled              yes
i686 features enabled     yes
CMOV is fast              yes
EBX available             yes
EBP available             yes
debug symbols             yes
strip symbols             yes
optimize for size         no
optimizations             yes
static                    yes
shared                    no
postprocessing support    no
network support           yes
threading support         pthreads
safe bitstream reader     yes
texi2html enabled         no
perl enabled              yes
pod2man enabled           yes
makeinfo enabled          yes
makeinfo supports HTML    yes
xmllint enabled           yes

External libraries:
appkit                  libxcb                  sdl2
avfoundation            libxcb_shape            securetransport
bzlib                   libxcb_shm              zlib
coreimage               libxcb_xfixes
iconv                   lzma

External libraries providing hardware acceleration:
audiotoolbox            videotoolbox

Libraries:
avcodec                 avformat                swscale
avdevice                avutil
avfilter                swresample

Programs:
ffmpeg                  ffplay                  ffprobe

Enabled decoders:
aac                     ffv1                    pcm_u8
aac_at                  ffvhuff                 pcm_vidc
aac_fixed               ffwavesynth             pcx
aac_latm                fic                     pfm
aasc                    fits                    pgm
ac3                     flac                    pgmyuv
ac3_at                  flashsv                 pgssub
ac3_fixed               flashsv2                pgx
acelp_kelvin            flic                    photocd
adpcm_4xm               flv                     pictor
adpcm_adx               fmvc                    pixlet
adpcm_afc               fourxm                  pjs
adpcm_agm               fraps                   png
adpcm_aica              frwu                    ppm
adpcm_argo              g2m                     prores
adpcm_ct                g723_1                  prosumer
adpcm_dtk               g729                    psd
adpcm_ea                gdv                     ptx
adpcm_ea_maxis_xa       gem                     qcelp
adpcm_ea_r1             gif                     qdm2
adpcm_ea_r2             gremlin_dpcm            qdm2_at
adpcm_ea_r3             gsm                     qdmc
adpcm_ea_xas            gsm_ms                  qdmc_at
adpcm_g722              gsm_ms_at               qdraw
adpcm_g726              h261                    qpeg
adpcm_g726le            h263                    qtrle
adpcm_ima_acorn         h263i                   r10k
adpcm_ima_alp           h263p                   r210
adpcm_ima_amv           h264                    ra_144
adpcm_ima_apc           hap                     ra_288
adpcm_ima_apm           hca                     ralf
adpcm_ima_cunning       hcom                    rasc
adpcm_ima_dat4          hevc                    rawvideo
adpcm_ima_dk3           hnm4_video              realtext
adpcm_ima_dk4           hq_hqa                  rl2
adpcm_ima_ea_eacs       hqx                     roq
adpcm_ima_ea_sead       huffyuv                 roq_dpcm
adpcm_ima_iss           hymt                    rpza
adpcm_ima_moflex        iac                     rscc
adpcm_ima_mtf           idcin                   rv10
adpcm_ima_oki           idf                     rv20
adpcm_ima_qt            iff_ilbm                rv30
adpcm_ima_qt_at         ilbc                    rv40
adpcm_ima_rad           ilbc_at                 s302m
adpcm_ima_smjpeg        imc                     sami
adpcm_ima_ssi           imm4                    sanm
adpcm_ima_wav           imm5                    sbc
adpcm_ima_ws            indeo2                  scpr
adpcm_ms                indeo3                  screenpresso
adpcm_mtaf              indeo4                  sdx2_dpcm
adpcm_psx               indeo5                  sga
adpcm_sbpro_2           interplay_acm           sgi
adpcm_sbpro_3           interplay_dpcm          sgirle
adpcm_sbpro_4           interplay_video         sheervideo
adpcm_swf               ipu                     shorten
adpcm_thp               jacosub                 simbiosis_imx
adpcm_thp_le            jpeg2000                sipr
adpcm_vima              jpegls                  siren
adpcm_xa                jv                      smackaud
adpcm_yamaha            kgv1                    smacker
adpcm_zork              kmvc                    smc
agm                     lagarith                smvjpeg
aic                     loco                    snow
alac                    lscr                    sol_dpcm
alac_at                 m101                    sonic
alias_pix               mace3                   sp5x
als                     mace6                   speedhq
amr_nb_at               magicyuv                speex
amrnb                   mdec                    srgc
amrwb                   metasound               srt
amv                     microdvd                ssa
anm                     mimic                   stl
ansi                    mjpeg                   subrip
ape                     mjpegb                  subviewer
apng                    mlp                     subviewer1
aptx                    mmvideo                 sunrast
aptx_hd                 mobiclip                svq1
arbc                    motionpixels            svq3
argo                    movtext                 tak
ass                     mp1                     targa
asv1                    mp1_at                  targa_y216
asv2                    mp1float                tdsc
atrac1                  mp2                     text
atrac3                  mp2_at                  theora
atrac3al                mp2float                thp
atrac3p                 mp3                     tiertexseqvideo
atrac3pal               mp3_at                  tiff
atrac9                  mp3adu                  tmv
aura                    mp3adufloat             truehd
aura2                   mp3float                truemotion1
av1                     mp3on4                  truemotion2
avrn                    mp3on4float             truemotion2rt
avrp                    mpc7                    truespeech
avs                     mpc8                    tscc
avui                    mpeg1video              tscc2
ayuv                    mpeg2video              tta
bethsoftvid             mpeg4                   twinvq
bfi                     mpegvideo               txd
bink                    mpl2                    ulti
binkaudio_dct           msa1                    utvideo
binkaudio_rdft          mscc                    v210
bintext                 msmpeg4v1               v210x
bitpacked               msmpeg4v2               v308
bmp                     msmpeg4v3               v408
bmv_audio               msnsiren                v410
bmv_video               msp2                    vb
brender_pix             msrle                   vble
c93                     mss1                    vc1
cavs                    mss2                    vc1image
ccaption                msvideo1                vcr1
cdgraphics              mszh                    vmdaudio
cdtoons                 mts2                    vmdvideo
cdxl                    mv30                    vmnc
cfhd                    mvc1                    vorbis
cinepak                 mvc2                    vp3
clearvideo              mvdv                    vp4
cljr                    mvha                    vp5
cllc                    mwsc                    vp6
comfortnoise            mxpeg                   vp6a
cook                    nellymoser              vp6f
cpia                    notchlc                 vp7
cri                     nuv                     vp8
cscd                    on2avc                  vp9
cyuv                    opus                    vplayer
dca                     paf_audio               vqa
dds                     paf_video               wavpack
derf_dpcm               pam                     wcmv
dfa                     pbm                     webp
dfpwm                   pcm_alaw                webvtt
dirac                   pcm_alaw_at             wmalossless
dnxhd                   pcm_bluray              wmapro
dolby_e                 pcm_dvd                 wmav1
dpx                     pcm_f16le               wmav2
dsd_lsbf                pcm_f24le               wmavoice
dsd_lsbf_planar         pcm_f32be               wmv1
dsd_msbf                pcm_f32le               wmv2
dsd_msbf_planar         pcm_f64be               wmv3
dsicinaudio             pcm_f64le               wmv3image
dsicinvideo             pcm_lxf                 wnv1
dss_sp                  pcm_mulaw               wrapped_avframe
dst                     pcm_mulaw_at            ws_snd1
dvaudio                 pcm_s16be               xan_dpcm
dvbsub                  pcm_s16be_planar        xan_wc3
dvdsub                  pcm_s16le               xan_wc4
dvvideo                 pcm_s16le_planar        xbin
dxa                     pcm_s24be               xbm
dxtory                  pcm_s24daud             xface
dxv                     pcm_s24le               xl
eac3                    pcm_s24le_planar        xma1
eac3_at                 pcm_s32be               xma2
eacmv                   pcm_s32le               xpm
eamad                   pcm_s32le_planar        xsub
eatgq                   pcm_s64be               xwd
eatgv                   pcm_s64le               y41p
eatqi                   pcm_s8                  ylc
eightbps                pcm_s8_planar           yop
eightsvx_exp            pcm_sga                 yuv4
eightsvx_fib            pcm_u16be               zero12v
escape124               pcm_u16le               zerocodec
escape130               pcm_u24be               zlib
evrc                    pcm_u24le               zmbv
exr                     pcm_u32be
fastaudio               pcm_u32le

Enabled encoders:
a64multi                h263p                   pgmyuv
a64multi5               h264_videotoolbox       png
aac                     hevc_videotoolbox       ppm
aac_at                  huffyuv                 prores
ac3                     ilbc_at                 prores_aw
ac3_fixed               jpeg2000                prores_ks
adpcm_adx               jpegls                  prores_videotoolbox
adpcm_argo              ljpeg                   qtrle
adpcm_g722              magicyuv                r10k
adpcm_g726              mjpeg                   r210
adpcm_g726le            mlp                     ra_144
adpcm_ima_alp           movtext                 rawvideo
adpcm_ima_amv           mp2                     roq
adpcm_ima_apm           mp2fixed                roq_dpcm
adpcm_ima_qt            mpeg1video              rpza
adpcm_ima_ssi           mpeg2video              rv10
adpcm_ima_wav           mpeg4                   rv20
adpcm_ima_ws            msmpeg4v2               s302m
adpcm_ms                msmpeg4v3               sbc
adpcm_swf               msvideo1                sgi
adpcm_yamaha            nellymoser              smc
alac                    opus                    snow
alac_at                 pam                     sonic
alias_pix               pbm                     sonic_ls
amv                     pcm_alaw                speedhq
apng                    pcm_alaw_at             srt
aptx                    pcm_bluray              ssa
aptx_hd                 pcm_dvd                 subrip
ass                     pcm_f32be               sunrast
asv1                    pcm_f32le               svq1
asv2                    pcm_f64be               targa
avrp                    pcm_f64le               text
avui                    pcm_mulaw               tiff
ayuv                    pcm_mulaw_at            truehd
bitpacked               pcm_s16be               tta
bmp                     pcm_s16be_planar        ttml
cfhd                    pcm_s16le               utvideo
cinepak                 pcm_s16le_planar        v210
cljr                    pcm_s24be               v308
comfortnoise            pcm_s24daud             v408
dca                     pcm_s24le               v410
dfpwm                   pcm_s24le_planar        vc2
dnxhd                   pcm_s32be               vorbis
dpx                     pcm_s32le               wavpack
dvbsub                  pcm_s32le_planar        webvtt
dvdsub                  pcm_s64be               wmav1
dvvideo                 pcm_s64le               wmav2
eac3                    pcm_s8                  wmv1
exr                     pcm_s8_planar           wmv2
ffv1                    pcm_u16be               wrapped_avframe
ffvhuff                 pcm_u16le               xbm
fits                    pcm_u24be               xface
flac                    pcm_u24le               xsub
flashsv                 pcm_u32be               xwd
flashsv2                pcm_u32le               y41p
flv                     pcm_u8                  yuv4
g723_1                  pcm_vidc                zlib
gif                     pcx                     zmbv
h261                    pfm
h263                    pgm

Enabled hwaccels:
h263_videotoolbox       mpeg1_videotoolbox      prores_videotoolbox
h264_videotoolbox       mpeg2_videotoolbox      vp9_videotoolbox
hevc_videotoolbox       mpeg4_videotoolbox

Enabled parsers:
aac                     dvbsub                  mpegvideo
aac_latm                dvd_nav                 opus
ac3                     dvdsub                  png
adx                     flac                    pnm
amr                     g723_1                  rv30
av1                     g729                    rv40
avs2                    gif                     sbc
avs3                    gsm                     sipr
bmp                     h261                    tak
cavsvideo               h263                    vc1
cook                    h264                    vorbis
cri                     hevc                    vp3
dca                     ipu                     vp8
dirac                   jpeg2000                vp9
dnxhd                   mjpeg                   webp
dolby_e                 mlp                     xbm
dpx                     mpeg4video              xma
dvaudio                 mpegaudio

Enabled demuxers:
aa                      hnm                     pcm_mulaw
aac                     ico                     pcm_s16be
aax                     idcin                   pcm_s16le
ac3                     idf                     pcm_s24be
ace                     iff                     pcm_s24le
acm                     ifv                     pcm_s32be
act                     ilbc                    pcm_s32le
adf                     image2                  pcm_s8
adp                     image2_alias_pix        pcm_u16be
ads                     image2_brender_pix      pcm_u16le
adx                     image2pipe              pcm_u24be
aea                     image_bmp_pipe          pcm_u24le
afc                     image_cri_pipe          pcm_u32be
aiff                    image_dds_pipe          pcm_u32le
aix                     image_dpx_pipe          pcm_u8
alp                     image_exr_pipe          pcm_vidc
amr                     image_gem_pipe          pjs
amrnb                   image_gif_pipe          pmp
amrwb                   image_j2k_pipe          pp_bnk
anm                     image_jpeg_pipe         pva
apc                     image_jpegls_pipe       pvf
ape                     image_pam_pipe          qcp
apm                     image_pbm_pipe          r3d
apng                    image_pcx_pipe          rawvideo
aptx                    image_pgm_pipe          realtext
aptx_hd                 image_pgmyuv_pipe       redspark
aqtitle                 image_pgx_pipe          rl2
argo_asf                image_photocd_pipe      rm
argo_brp                image_pictor_pipe       roq
argo_cvg                image_png_pipe          rpl
asf                     image_ppm_pipe          rsd
asf_o                   image_psd_pipe          rso
ass                     image_qdraw_pipe        rtp
ast                     image_sgi_pipe          rtsp
au                      image_sunrast_pipe      s337m
av1                     image_svg_pipe          sami
avi                     image_tiff_pipe         sap
avr                     image_webp_pipe         sbc
avs                     image_xbm_pipe          sbg
avs2                    image_xpm_pipe          scc
avs3                    image_xwd_pipe          scd
bethsoftvid             ingenient               sdp
bfi                     ipmovie                 sdr2
bfstm                   ipu                     sds
bink                    ircam                   sdx
binka                   iss                     segafilm
bintext                 iv8                     ser
bit                     ivf                     sga
bitpacked               ivr                     shorten
bmv                     jacosub                 siff
boa                     jv                      simbiosis_imx
brstm                   kux                     sln
c93                     kvag                    smacker
caf                     live_flv                smjpeg
cavsvideo               lmlm4                   smush
cdg                     loas                    sol
cdxl                    lrc                     sox
cine                    luodat                  spdif
codec2                  lvf                     srt
codec2raw               lxf                     stl
concat                  m4v                     str
data                    matroska                subviewer
daud                    mca                     subviewer1
dcstr                   mcc                     sup
derf                    mgsts                   svag
dfa                     microdvd                svs
dfpwm                   mjpeg                   swf
dhav                    mjpeg_2000              tak
dirac                   mlp                     tedcaptions
dnxhd                   mlv                     thp
dsf                     mm                      threedostr
dsicin                  mmf                     tiertexseq
dss                     mods                    tmv
dts                     moflex                  truehd
dtshd                   mov                     tta
dv                      mp3                     tty
dvbsub                  mpc                     txd
dvbtxt                  mpc8                    ty
dxa                     mpegps                  v210
ea                      mpegts                  v210x
ea_cdata                mpegtsraw               vag
eac3                    mpegvideo               vc1
epaf                    mpjpeg                  vc1t
ffmetadata              mpl2                    vividas
filmstrip               mpsub                   vivo
fits                    msf                     vmd
flac                    msnwc_tcp               vobsub
flic                    msp                     voc
flv                     mtaf                    vpk
fourxm                  mtv                     vplayer
frm                     musx                    vqf
fsb                     mv                      w64
fwse                    mvi                     wav
g722                    mxf                     wc3
g723_1                  mxg                     webm_dash_manifest
g726                    nc                      webvtt
g726le                  nistsphere              wsaud
g729                    nsp                     wsd
gdv                     nsv                     wsvqa
genh                    nut                     wtv
gif                     nuv                     wv
gsm                     obu                     wve
gxf                     ogg                     xa
h261                    oma                     xbin
h263                    paf                     xmv
h264                    pcm_alaw                xvag
hca                     pcm_f32be               xwma
hcom                    pcm_f32le               yop
hevc                    pcm_f64be               yuv4mpegpipe
hls                     pcm_f64le

Enabled muxers:
a64                     h264                    pcm_s24be
ac3                     hash                    pcm_s24le
adts                    hds                     pcm_s32be
adx                     hevc                    pcm_s32le
aiff                    hls                     pcm_s8
alp                     ico                     pcm_u16be
amr                     ilbc                    pcm_u16le
amv                     image2                  pcm_u24be
apm                     image2pipe              pcm_u24le
apng                    ipod                    pcm_u32be
aptx                    ircam                   pcm_u32le
aptx_hd                 ismv                    pcm_u8
argo_asf                ivf                     pcm_vidc
argo_cvg                jacosub                 psp
asf                     kvag                    rawvideo
asf_stream              latm                    rm
ass                     lrc                     roq
ast                     m4v                     rso
au                      matroska                rtp
avi                     matroska_audio          rtp_mpegts
avm2                    md5                     rtsp
avs2                    microdvd                sap
avs3                    mjpeg                   sbc
bit                     mkvtimestamp_v2         scc
caf                     mlp                     segafilm
cavsvideo               mmf                     segment
codec2                  mov                     smjpeg
codec2raw               mp2                     smoothstreaming
crc                     mp3                     sox
dash                    mp4                     spdif
data                    mpeg1system             spx
daud                    mpeg1vcd                srt
dfpwm                   mpeg1video              stream_segment
dirac                   mpeg2dvd                streamhash
dnxhd                   mpeg2svcd               sup
dts                     mpeg2video              swf
dv                      mpeg2vob                tee
eac3                    mpegts                  tg2
f4v                     mpjpeg                  tgp
ffmetadata              mxf                     truehd
fifo                    mxf_d10                 tta
fifo_test               mxf_opatom              ttml
filmstrip               null                    uncodedframecrc
fits                    nut                     vc1
flac                    obu                     vc1t
flv                     oga                     voc
framecrc                ogg                     w64
framehash               ogv                     wav
framemd5                oma                     webm
g722                    opus                    webm_chunk
g723_1                  pcm_alaw                webm_dash_manifest
g726                    pcm_f32be               webp
g726le                  pcm_f32le               webvtt
gif                     pcm_f64be               wsaud
gsm                     pcm_f64le               wtv
gxf                     pcm_mulaw               wv
h261                    pcm_s16be               yuv4mpegpipe
h263                    pcm_s16le

Enabled protocols:
async                   http                    rtmpts
cache                   httpproxy               rtp
concat                  https                   srtp
concatf                 icecast                 subfile
crypto                  md5                     tcp
data                    mmsh                    tee
ffrtmphttp              mmst                    tls
file                    pipe                    udp
ftp                     prompeg                 udplite
gopher                  rtmp                    unix
gophers                 rtmps
hls                     rtmpt

Enabled filters:
abench                  copy                    negate
abitscope               coreimage               nlmeans
acompressor             coreimagesrc            noformat
acontrast               crop                    noise
acopy                   crossfeed               normalize
acrossfade              crystalizer             null
acrossover              cue                     nullsink
acrusher                curves                  nullsrc
acue                    datascope               oscilloscope
addroi                  dblur                   overlay
adeclick                dcshift                 pad
adeclip                 dctdnoiz                pal100bars
adecorrelate            deband                  pal75bars
adelay                  deblock                 palettegen
adenorm                 decimate                paletteuse
aderivative             deconvolve              pan
adrawgraph              dedot                   perms
adynamicequalizer       deesser                 photosensitivity
adynamicsmooth          deflate                 pixdesctest
aecho                   deflicker               pixscope
aemphasis               dejudder                premultiply
aeval                   derain                  prewitt
aevalsrc                deshake                 pseudocolor
aexciter                despill                 psnr
afade                   detelecine              qp
afftdn                  dialoguenhance          random
afftfilt                dilation                readeia608
afifo                   displace                readvitc
afir                    dnn_classify            realtime
afirsrc                 dnn_detect              remap
aformat                 dnn_processing          removegrain
afreqshift              doubleweave             removelogo
afwtdn                  drawbox                 replaygain
agate                   drawgraph               reverse
agraphmonitor           drawgrid                rgbashift
ahistogram              drmeter                 rgbtestsrc
aiir                    dynaudnorm              roberts
aintegral               earwax                  rotate
ainterleave             ebur128                 scale
alatency                edgedetect              scale2ref
alimiter                elbg                    scdet
allpass                 entropy                 scharr
allrgb                  epx                     scroll
allyuv                  equalizer               segment
aloop                   erosion                 select
alphaextract            estdif                  selectivecolor
alphamerge              exposure                sendcmd
amerge                  extractplanes           separatefields
ametadata               extrastereo             setdar
amix                    fade                    setfield
amovie                  fftdnoiz                setparams
amplify                 fftfilt                 setpts
amultiply               field                   setrange
anequalizer             fieldhint               setsar
anlmdn                  fieldmatch              settb
anlmf                   fieldorder              shear
anlms                   fifo                    showcqt
anoisesrc               fillborders             showfreqs
anull                   firequalizer            showinfo
anullsink               flanger                 showpalette
anullsrc                floodfill               showspatial
apad                    format                  showspectrum
aperms                  fps                     showspectrumpic
aphasemeter             framepack               showvolume
aphaser                 framerate               showwaves
aphaseshift             framestep               showwavespic
apsyclip                freezedetect            shuffleframes
apulsator               freezeframes            shufflepixels
arealtime               gblur                   shuffleplanes
aresample               geq                     sidechaincompress
areverse                gradfun                 sidechaingate
arnndn                  gradients               sidedata
asdr                    graphmonitor            sierpinski
asegment                grayworld               signalstats
aselect                 greyedge                silencedetect
asendcmd                guided                  silenceremove
asetnsamples            haas                    sinc
asetpts                 haldclut                sine
asetrate                haldclutsrc             smptebars
asettb                  hdcd                    smptehdbars
ashowinfo               headphone               sobel
asidedata               hflip                   spectrumsynth
asoftclip               highpass                speechnorm
aspectralstats          highshelf               split
asplit                  hilbert                 sr
astats                  histogram               ssim
astreamselect           hqx                     stereotools
asubboost               hstack                  stereowiden
asubcut                 hsvhold                 streamselect
asupercut               hsvkey                  superequalizer
asuperpass              hue                     surround
asuperstop              huesaturation           swaprect
atadenoise              hwdownload              swapuv
atempo                  hwmap                   tblend
atilt                   hwupload                telecine
atrim                   hysteresis              testsrc
avectorscope            identity                testsrc2
avgblur                 idet                    thistogram
axcorrelate             il                      threshold
bandpass                inflate                 thumbnail
bandreject              interleave              tile
bass                    join                    tlut2
bbox                    kirsch                  tmedian
bench                   lagfun                  tmidequalizer
bilateral               latency                 tmix
biquad                  lenscorrection          tonemap
bitplanenoise           life                    tpad
blackdetect             limitdiff               transpose
blend                   limiter                 treble
bm3d                    loop                    tremolo
bwdif                   loudnorm                trim
cas                     lowpass                 unpremultiply
cellauto                lowshelf                unsharp
channelmap              lumakey                 untile
channelsplit            lut                     v360
chorus                  lut1d                   varblur
chromahold              lut2                    vectorscope
chromakey               lut3d                   vflip
chromanr                lutrgb                  vfrdet
chromashift             lutyuv                  vibrance
ciescope                mandelbrot              vibrato
codecview               maskedclamp             vif
color                   maskedmax               vignette
colorbalance            maskedmerge             vmafmotion
colorchannelmixer       maskedmin               volume
colorcontrast           maskedthreshold         volumedetect
colorcorrect            maskfun                 vstack
colorhold               mcompand                w3fdif
colorize                median                  waveform
colorkey                mergeplanes             weave
colorlevels             mestimate               xbr
colorspace              metadata                xcorrelate
colorspectrum           midequalizer            xfade
colortemperature        minterpolate            xmedian
compand                 mix                     xstack
compensationdelay       monochrome              yadif
concat                  morpho                  yaepblur
convolution             movie                   yuvtestsrc
convolve                msad                    zoompan

Enabled bsfs:
aac_adtstoasc           h264_redundant_pps      opus_metadata
av1_frame_merge         hapqa_extract           pcm_rechunk
av1_frame_split         hevc_metadata           prores_metadata
av1_metadata            hevc_mp4toannexb        remove_extradata
chomp                   imx_dump_header         setts
dca_core                mjpeg2jpeg              text2movsub
dump_extradata          mjpega_dump_header      trace_headers
dv_error_marker         mov2textsub             truehd_core
eac3_core               mp3_header_decompress   vp9_metadata
extract_extradata       mpeg2_metadata          vp9_raw_reorder
filter_units            mpeg4_unpack_bframes    vp9_superframe
h264_metadata           noise                   vp9_superframe_split
h264_mp4toannexb        null

Enabled indevs:
avfoundation            lavfi                   xcbgrab

Enabled outdevs:
audiotoolbox            sdl2

License: LGPL version 2.1 or later
(base) HaoyeMacBookPro:ffmpeg kancheng$ 

```

2. 編譯成功


```
(base) HaoyeMacBookPro:ffmpeg kancheng$ ffmpeg
ffmpeg version 5.0 Copyright (c) 2000-2022 the FFmpeg developers
  built with Apple clang version 13.0.0 (clang-1300.0.29.30)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/5.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox
  libavutil      57. 17.100 / 57. 17.100
  libavcodec     59. 18.100 / 59. 18.100
  libavformat    59. 16.100 / 59. 16.100
  libavdevice    59.  4.100 / 59.  4.100
  libavfilter     8. 24.100 /  8. 24.100
  libswscale      6.  4.100 /  6.  4.100
  libswresample   4.  3.100 /  4.  3.100
  libpostproc    56.  3.100 / 56.  3.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

Use -h to get full help or, even better, run 'man ffmpeg'
```


```
(base) HaoyeMacBookPro:ffmpeg kancheng$ ffmpeg -h
ffmpeg version 5.0 Copyright (c) 2000-2022 the FFmpeg developers
  built with Apple clang version 13.0.0 (clang-1300.0.29.30)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/5.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox
  libavutil      57. 17.100 / 57. 17.100
  libavcodec     59. 18.100 / 59. 18.100
  libavformat    59. 16.100 / 59. 16.100
  libavdevice    59.  4.100 / 59.  4.100
  libavfilter     8. 24.100 /  8. 24.100
  libswscale      6.  4.100 /  6.  4.100
  libswresample   4.  3.100 /  4.  3.100
  libpostproc    56.  3.100 / 56.  3.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

Getting help:
    -h      -- print basic options
    -h long -- print more options
    -h full -- print all options (including all format and codec specific options, very long)
    -h type=name -- print all options for the named decoder/encoder/demuxer/muxer/filter/bsf/protocol
    See man ffmpeg for detailed description of the options.

Print help / information / capabilities:
-L                  show license
-h topic            show help
-? topic            show help
-help topic         show help
--help topic        show help
-version            show version
-buildconf          show build configuration
-formats            show available formats
-muxers             show available muxers
-demuxers           show available demuxers
-devices            show available devices
-codecs             show available codecs
-decoders           show available decoders
-encoders           show available encoders
-bsfs               show available bit stream filters
-protocols          show available protocols
-filters            show available filters
-pix_fmts           show available pixel formats
-layouts            show standard channel layouts
-sample_fmts        show available audio sample formats
-dispositions       show available stream dispositions
-colors             show available color names
-sources device     list sources of the input device
-sinks device       list sinks of the output device
-hwaccels           show available HW acceleration methods

Global options (affect whole program instead of just one file):
-loglevel loglevel  set logging level
-v loglevel         set logging level
-report             generate a report
-max_alloc bytes    set maximum size of a single allocated block
-y                  overwrite output files
-n                  never overwrite output files
-ignore_unknown     Ignore unknown stream types
-filter_threads     number of non-complex filter threads
-filter_complex_threads  number of threads for -filter_complex
-stats              print progress report during encoding
-max_error_rate maximum error rate  ratio of decoding errors (0.0: no errors, 1.0: 100% errors) above which ffmpeg returns an error instead of success.
-vol volume         change audio volume (256=normal)

Per-file main options:
-f fmt              force format
-c codec            codec name
-codec codec        codec name
-pre preset         preset name
-map_metadata outfile[,metadata]:infile[,metadata]  set metadata information of outfile from infile
-t duration         record or transcode "duration" seconds of audio/video
-to time_stop       record or transcode stop time
-fs limit_size      set the limit file size in bytes
-ss time_off        set the start time offset
-sseof time_off     set the start time offset relative to EOF
-seek_timestamp     enable/disable seeking by timestamp with -ss
-timestamp time     set the recording timestamp ('now' to set the current time)
-metadata string=string  add metadata
-program title=string:st=number...  add program with specified streams
-target type        specify target file type ("vcd", "svcd", "dvd", "dv" or "dv50" with optional prefixes "pal-", "ntsc-" or "film-")
-apad               audio pad
-frames number      set the number of frames to output
-filter filter_graph  set stream filtergraph
-filter_script filename  read stream filtergraph description from a file
-reinit_filter      reinit filtergraph on input parameter changes
-discard            discard
-disposition        disposition

Video options:
-vframes number     set the number of video frames to output
-r rate             set frame rate (Hz value, fraction or abbreviation)
-fpsmax rate        set max frame rate (Hz value, fraction or abbreviation)
-s size             set frame size (WxH or abbreviation)
-aspect aspect      set aspect ratio (4:3, 16:9 or 1.3333, 1.7777)
-vn                 disable video
-vcodec codec       force video codec ('copy' to copy stream)
-timecode hh:mm:ss[:;.]ff  set initial TimeCode value.
-pass n             select the pass number (1 to 3)
-vf filter_graph    set video filters
-ab bitrate         audio bitrate (please use -b:a)
-b bitrate          video bitrate (please use -b:v)
-dn                 disable data

Audio options:
-aframes number     set the number of audio frames to output
-aq quality         set audio quality (codec-specific)
-ar rate            set audio sampling rate (in Hz)
-ac channels        set number of audio channels
-an                 disable audio
-acodec codec       force audio codec ('copy' to copy stream)
-vol volume         change audio volume (256=normal)
-af filter_graph    set audio filters

Subtitle options:
-s size             set frame size (WxH or abbreviation)
-sn                 disable subtitle
-scodec codec       force subtitle codec ('copy' to copy stream)
-stag fourcc/tag    force subtitle tag/fourcc
-fix_sub_duration   fix subtitles duration
-canvas_size size   set canvas size (WxH or abbreviation)
-spre preset        set the subtitle options to the indicated preset


(base) HaoyeMacBookPro:ffmpeg kancheng$ 
```

