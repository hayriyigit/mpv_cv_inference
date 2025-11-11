#!/bin/bash
VIDEO_FILE=/home/ai/Documents/Desktop/D4_TCJTE_media_record_2024-02-20_08-45-45_1708407924_D4_raw.ts
MULTICAST_ADDR=239.255.0.1:11001
LOOP_OPTS="-re -stream_loop -1"
STREAM_OPTS="-flags +global_header -c copy -an"
CMD="ffmpeg $LOOP_OPTS -i $VIDEO_FILE $STREAM_OPTS -f mpegts udp://${MULTICAST_ADDR}?pkt_size=1316"
echo $CMD
$CMD