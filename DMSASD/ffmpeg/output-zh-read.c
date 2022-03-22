/*
 * Copyright (c) 2003 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/* ZH *
版權所有 (c) 2003 Fabrice Bellard
 
特此免費授予任何獲得本軟件和相關文檔文件（“軟件”）副本的人，以不受限制地處理本軟件，包括但不限於使用、複製、修改、合併的權利、發布、分發、再許可和/或出售本軟件的副本，並允許向其提供本軟件的人這樣做，但須符合以下條件：

上述版權聲明和本許可聲明應包含在本軟件的所有副本或大部分內容中。

本軟件按“原樣”提供，不提供任何形式的明示或暗示保證，包括但不限於適銷性、特定用途適用性和非侵權保證。

在任何情況下，作者或版權持有人均不對任何索賠、損害或其他責任承擔任何責任，無論是在合同、侵權或其他方面，由本軟件或本軟件的使用或其他交易引起或與之相關。軟件。
*/

/**
 * @file
 * libavformat API example.
 *
 * @example doc/examples/output.c
 * Output a media file in any supported libavformat format.
 * The default codecs are used.
 */

/* ZH *
@文件
libavformat API 示例。

@example doc/examples/output.c
以任何受支持的 libavformat 格式輸出媒體文件。
使用默認編解碼器。
*/

// C Library

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// FFMPEG Library
#include "libavutil/mathematics.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"


// 巨集定義
/* 5 seconds stream duration */
#define STREAM_DURATION   5.0
#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */

static int sws_flags = SWS_BICUBIC;

/**************************************************************/
/* audio output */
// 音源輸出
static float t, tincr, tincr2;
static int16_t *samples;
static int audio_input_frame_size;

/* 加入音源輸出
 * add an audio output stream
 */
static AVStream *add_audio_stream(AVFormatContext *oc, enum AVCodecID codec_id)
{
    AVCodecContext *c;
    AVStream *st;
    AVCodec *codec;
// 找到音頻編碼器
    /* find the audio encoder */
    codec = avcodec_find_encoder(codec_id);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    st = avformat_new_stream(oc, codec);
    if (!st) {
        fprintf(stderr, "Could not alloc stream\n");
        exit(1);
    }

    c = st->codec;

// 放樣本參數
    /* put sample parameters */
    c->sample_fmt  = AV_SAMPLE_FMT_S16;
    c->bit_rate    = 64000;
    c->sample_rate = 44100;
    c->channels    = 2;

// 某些格式希望 stream header 是分開的
    // some formats want stream headers to be separate
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

    return st;
}

// 開啟音源
static void open_audio(AVFormatContext *oc, AVStream *st)
{
    AVCodecContext *c;

    c = st->codec;

    /* open it */
    if (avcodec_open2(c, NULL, NULL) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }
// 初始化信號發生器
    /* init signal generator */
    t     = 0;
    tincr = 2 * M_PI * 110.0 / c->sample_rate;
// 以每秒 110 Hz 的速度遞增頻率
    /* increment frequency by 110 Hz per second */
    tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;

    if (c->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE)
        audio_input_frame_size = 10000;
    else
        audio_input_frame_size = c->frame_size;
    samples = av_malloc(audio_input_frame_size *
                        av_get_bytes_per_sample(c->sample_fmt) *
                        c->channels);
}

// 準備“frame_size”樣本的 16 位虛擬音頻幀和 'nb_channels' 頻道。

/* Prepare a 16 bit dummy audio frame of 'frame_size' samples and
 * 'nb_channels' channels. */
static void get_audio_frame(int16_t *samples, int frame_size, int nb_channels)
{
    int j, i, v;
    int16_t *q;

    q = samples;
    for (j = 0; j < frame_size; j++) {
        v = (int)(sin(t) * 10000);
        for (i = 0; i < nb_channels; i++)
            *q++ = v;
        t     += tincr;
        tincr += tincr2;
    }
}

static void write_audio_frame(AVFormatContext *oc, AVStream *st)
{
    AVCodecContext *c;
    AVPacket pkt = { 0 }; // data and size must be 0; 數據和大小必須為 0；
    AVFrame *frame = av_frame_alloc();
    int got_packet;

    av_init_packet(&pkt);
    c = st->codec;

    get_audio_frame(samples, audio_input_frame_size, c->channels);
    frame->nb_samples = audio_input_frame_size;
    avcodec_fill_audio_frame(frame, c->channels, c->sample_fmt,
                             (uint8_t *)samples,
                             audio_input_frame_size *
                             av_get_bytes_per_sample(c->sample_fmt) *
                             c->channels, 1);

    avcodec_encode_audio2(c, &pkt, frame, &got_packet);
    if (!got_packet)
        return;

    pkt.stream_index = st->index;
// 將壓縮幀寫入媒體文件。
    /* Write the compressed frame to the media file. */
    if (av_interleaved_write_frame(oc, &pkt) != 0) {
        fprintf(stderr, "Error while writing audio frame\n");
        exit(1);
    }
    avcodec_free_frame(&frame);
}

static void close_audio(AVFormatContext *oc, AVStream *st)
{
    avcodec_close(st->codec);

    av_free(samples);
}

/**************************************************************/
/* video output */
// 影像輸出

static AVFrame *picture, *tmp_picture;
static int frame_count;

// 加入影像輸出的 Stream
/* Add a video output stream. */
static AVStream *add_video_stream(AVFormatContext *oc, enum AVCodecID codec_id)
{
    AVCodecContext *c;
    AVStream *st;
    AVCodec *codec;
    // 找到影像的 encoder
    /* find the video encoder */
    codec = avcodec_find_encoder(codec_id);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    st = avformat_new_stream(oc, codec);
    if (!st) {
        fprintf(stderr, "Could not alloc stream\n");
        exit(1);
    }

    c = st->codec;

    // 放樣本參數。
    /* Put sample parameters. */
    c->bit_rate = 400000;
    // 分辨率必須是二的倍數。
    /* Resolution must be a multiple of two. */
    c->width    = 352;
    c->height   = 288;
    // 時基：這是表示幀時間戳的基本時間單位（以秒為單位）。 對於固定 fps 內容，時基應為 1/幀速率，時間戳增量應等於 1。
    /* timebase: This is the fundamental unit of time (in seconds) in terms
     * of which frame timestamps are represented. For fixed-fps content,
     * timebase should be 1/framerate and timestamp increments should be
     * identical to 1. */
    c->time_base.den = STREAM_FRAME_RATE;
    c->time_base.num = 1;
    // 最多每十二幀發射一幀
    c->gop_size      = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt       = STREAM_PIX_FMT;
    if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
        // 只是為了測試，我們還添加了B幀
        /* just for testing, we also add B frames */
        c->max_b_frames = 2;
    }
    if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
        /* Needed to avoid using macroblocks in which some coeffs overflow.
         * This does not happen with normal video, it just happens here as
         * the motion of the chroma plane does not match the luma plane. */
        /* 需要避免使用某些係數溢出的宏塊。
這不會發生在普通視頻中，它只是在這裡發生，因為色度平面的運動與亮度平面不匹配。 */
        c->mb_decision = 2;
    }
    // 某些格式希望流標頭是分開的。
    /* Some formats want stream headers to be separate. */
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

    return st;
}

static AVFrame *alloc_picture(enum AVPixelFormat pix_fmt, int width, int height)
{
    AVFrame *picture;
    uint8_t *picture_buf;
    int size;

    picture = av_frame_alloc();
    if (!picture)
        return NULL;
    size        = avpicture_get_size(pix_fmt, width, height);
    picture_buf = av_malloc(size);
    if (!picture_buf) {
        av_free(picture);
        return NULL;
    }
    avpicture_fill((AVPicture *)picture, picture_buf,
                   pix_fmt, width, height);
    return picture;
}

static void open_video(AVFormatContext *oc, AVStream *st)
{
    AVCodecContext *c;

    c = st->codec;
    // 開啟
    /* open the codec */
    if (avcodec_open2(c, NULL, NULL) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }
    // 分配編碼的原始圖片。
    /* Allocate the encoded raw picture. */
    picture = alloc_picture(c->pix_fmt, c->width, c->height);
    if (!picture) {
        fprintf(stderr, "Could not allocate picture\n");
        exit(1);
    }

    /* If the output format is not YUV420P, then a temporary YUV420P
     * picture is needed too. It is then converted to the required
     * output format. */
    // 如果輸出格式不是 YUV420P，那麼也需要一張臨時的 YUV420P 圖片。 然後將其轉換為所需的輸出格式。
    tmp_picture = NULL;
    if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
        tmp_picture = alloc_picture(AV_PIX_FMT_YUV420P, c->width, c->height);
        if (!tmp_picture) {
            fprintf(stderr, "Could not allocate temporary picture\n");
            exit(1);
        }
    }
}
// 準備一個虛擬圖像。
/* Prepare a dummy image. */
static void fill_yuv_image(AVFrame *pict, int frame_index,
                           int width, int height)
{
    int x, y, i;

    i = frame_index;

    /* Y */
    for (y = 0; y < height; y++)
        for (x = 0; x < width; x++)
            pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;

    /* Cb and Cr */
    for (y = 0; y < height / 2; y++) {
        for (x = 0; x < width / 2; x++) {
            pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
            pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
        }
    }
}

static void write_video_frame(AVFormatContext *oc, AVStream *st)
{
    int ret;
    AVCodecContext *c;
    static struct SwsContext *img_convert_ctx;

    c = st->codec;

    if (frame_count >= STREAM_NB_FRAMES) {
        /* No more frames to compress. The codec has a latency of a few
         * frames if using B-frames, so we get the last frames by
         * passing the same picture again. */
// 不再需要壓縮幀。 如果使用 B 幀，編解碼器有幾幀的延遲，所以我們通過再次傳遞相同的圖片來獲得最後一幀。
    } else {
        if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
            /* as we only generate a YUV420P picture, we must convert it
             * to the codec pixel format if needed */
// 由於我們只生成一張 YUV420P 圖片，如果需要，我們必須將其轉換為編解碼器像素格式
            if (img_convert_ctx == NULL) {
                img_convert_ctx = sws_getContext(c->width, c->height,
                                                 AV_PIX_FMT_YUV420P,
                                                 c->width, c->height,
                                                 c->pix_fmt,
                                                 sws_flags, NULL, NULL, NULL);
                if (img_convert_ctx == NULL) {
                    fprintf(stderr,
                            "Cannot initialize the conversion context\n");
                    exit(1);
                }
            }
            fill_yuv_image(tmp_picture, frame_count, c->width, c->height);
            sws_scale(img_convert_ctx, tmp_picture->data, tmp_picture->linesize,
                      0, c->height, picture->data, picture->linesize);
        } else {
            fill_yuv_image(picture, frame_count, c->width, c->height);
        }
    }

    if (oc->oformat->flags & AVFMT_RAWPICTURE) {
// 原始視頻案例 - API 將在不久的將來略有變化。
        /* Raw video case - the API will change slightly in the near
         * future for that. */
        AVPacket pkt;
        av_init_packet(&pkt);

        pkt.flags        |= AV_PKT_FLAG_KEY;
        pkt.stream_index  = st->index;
        pkt.data          = (uint8_t *)picture;
        pkt.size          = sizeof(AVPicture);

        ret = av_interleaved_write_frame(oc, &pkt);
    } else {
        AVPacket pkt = { 0 };
        int got_packet;
        av_init_packet(&pkt);
        // encode 影像
        /* encode the image */
        ret = avcodec_encode_video2(c, &pkt, picture, &got_packet);
        /* If size is zero, it means the image was buffered. */
        // 如果大小為零，則表示圖像已緩衝。
        if (!ret && got_packet && pkt.size) {
            if (pkt.pts != AV_NOPTS_VALUE) {
                pkt.pts = av_rescale_q(pkt.pts,
                                       c->time_base, st->time_base);
            }
            if (pkt.dts != AV_NOPTS_VALUE) {
                pkt.dts = av_rescale_q(pkt.dts,
                                       c->time_base, st->time_base);
            }
            pkt.stream_index = st->index;
            // 將壓縮幀寫入媒體文件。
            /* Write the compressed frame to the media file. */
            ret = av_interleaved_write_frame(oc, &pkt);
        } else {
            ret = 0;
        }
    }
    if (ret != 0) {
        fprintf(stderr, "Error while writing video frame\n");
        exit(1);
    }
    frame_count++;
}
// 關閉
static void close_video(AVFormatContext *oc, AVStream *st)
{
    avcodec_close(st->codec);
    av_free(picture->data[0]);
    av_free(picture);
    if (tmp_picture) {
        av_free(tmp_picture->data[0]);
        av_free(tmp_picture);
    }
}

/**************************************************************/
/* media file output */
// 當檔案輸出
int main(int argc, char **argv)
{
    const char *filename;
    AVOutputFormat *fmt;
    AVFormatContext *oc;
    AVStream *audio_st, *video_st;
    double audio_pts, video_pts;
    int i;
// 初始化 libavcodec，並註冊所有編解碼器和格式。
    /* Initialize libavcodec, and register all codecs and formats. */
    av_register_all();

    if (argc != 2) {
        printf("usage: %s output_file\n"
               "API example program to output a media file with libavformat.\n"
               "The output format is automatically guessed according to the file extension.\n"
               "Raw images can also be output by using '%%d' in the filename\n"
               "\n", argv[0]);
        return 1;
    }

    filename = argv[1];
// 從名稱中自動檢測輸出格式。 默認為 MPEG。
    /* Autodetect the output format from the name. default is MPEG. */
    fmt = av_guess_format(NULL, filename, NULL);
    if (!fmt) {
        printf("Could not deduce output format from file extension: using MPEG.\n");
        fmt = av_guess_format("mpeg", NULL, NULL);
    }
    if (!fmt) {
        fprintf(stderr, "Could not find suitable output format\n");
        return 1;
    }
// 分配輸出媒體上下文。
    /* Allocate the output media context. */
    oc = avformat_alloc_context();
    if (!oc) {
        fprintf(stderr, "Memory error\n");
        return 1;
    }
    oc->oformat = fmt;
    snprintf(oc->filename, sizeof(oc->filename), "%s", filename);
// 使用默認格式編解碼器添加音頻和視頻流並初始化編解碼器。
    /* Add the audio and video streams using the default format codecs
     * and initialize the codecs. */
    video_st = NULL;
    audio_st = NULL;
    if (fmt->video_codec != AV_CODEC_ID_NONE) {
        video_st = add_video_stream(oc, fmt->video_codec);
    }
    if (fmt->audio_codec != AV_CODEC_ID_NONE) {
        audio_st = add_audio_stream(oc, fmt->audio_codec);
    }
// 現在所有參數都設置好了，我們可以打開音頻和視頻編解碼器並分配必要的編碼緩衝區。
    /* Now that all the parameters are set, we can open the audio and
     * video codecs and allocate the necessary encode buffers. */
    if (video_st)
        open_video(oc, video_st);
    if (audio_st)
        open_audio(oc, audio_st);

    av_dump_format(oc, 0, filename, 1);
// 如果需要，打開輸出文件
    /* open the output file, if needed */
    if (!(fmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&oc->pb, filename, AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Could not open '%s'\n", filename);
            return 1;
        }
    }
// 寫入流標頭（如果有）。
    /* Write the stream header, if any. */
    avformat_write_header(oc, NULL);

    for (;;) {
        // 計算當前的音頻和視頻時間。
        /* Compute current audio and video time. */
        if (audio_st)
            audio_pts = (double)audio_st->pts.val * audio_st->time_base.num / audio_st->time_base.den;
        else
            audio_pts = 0.0;

        if (video_st)
            video_pts = (double)video_st->pts.val * video_st->time_base.num /
                        video_st->time_base.den;
        else
            video_pts = 0.0;

        if ((!audio_st || audio_pts >= STREAM_DURATION) &&
            (!video_st || video_pts >= STREAM_DURATION))
            break;
// 寫入交錯的音頻和視頻幀
        /* write interleaved audio and video frames */
        if (!video_st || (video_st && audio_st && audio_pts < video_pts)) {
            write_audio_frame(oc, audio_st);
        } else {
            write_video_frame(oc, video_st);
        }
    }
// 寫預告片，如果有的話。 必須在您關閉編寫標頭時打開的 CodecContexts 之前編寫預告片； 否則 av_write_trailer() 可能會嘗試使用在 av_codec_close() 上釋放的內存。
    /* Write the trailer, if any. The trailer must be written before you
     * close the CodecContexts open when you wrote the header; otherwise
     * av_write_trailer() may try to use memory that was freed on
     * av_codec_close(). */
    av_write_trailer(oc);
// 關閉每一個 codec.
    /* Close each codec. */
    if (video_st)
        close_video(oc, video_st);
    if (audio_st)
        close_audio(oc, audio_st);

    /* Free the streams. */
    for (i = 0; i < oc->nb_streams; i++) {
        av_freep(&oc->streams[i]->codec);
        av_freep(&oc->streams[i]);
    }

    if (!(fmt->flags & AVFMT_NOFILE))
        // 關閉
        /* Close the output file. */
        avio_close(oc->pb);

    /* free the stream */
    av_free(oc);

    return 0;
}
