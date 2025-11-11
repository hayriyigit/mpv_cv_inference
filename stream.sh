#!/bin/bash

# UDP Video Streaming Script
# This script streams video over UDP using ffmpeg
# Multiple clients can connect and receive the stream

# Configuration
VIDEO_SOURCE="${1:-/dev/video0}"  # Default to webcam, or specify file path
UDP_HOST="${2:-239.0.0.1}"        # Multicast address (allows multiple readers)
UDP_PORT="${3:-1234}"
BITRATE="${4:-2M}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== UDP Video Streamer ===${NC}"
echo ""

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed${NC}"
    echo "Install with: sudo apt-get install ffmpeg (Ubuntu/Debian)"
    echo "           or: brew install ffmpeg (macOS)"
    exit 1
fi

# Determine if source is a file or device
if [[ -f "$VIDEO_SOURCE" ]]; then
    echo -e "${GREEN}Streaming from file: ${YELLOW}$VIDEO_SOURCE${NC}"
    INPUT_OPTIONS="-re -i $VIDEO_SOURCE"
elif [[ "$VIDEO_SOURCE" == /dev/video* ]]; then
    echo -e "${GREEN}Streaming from camera: ${YELLOW}$VIDEO_SOURCE${NC}"
    INPUT_OPTIONS="-f v4l2 -i $VIDEO_SOURCE"
else
    echo -e "${YELLOW}Assuming network stream or special source: $VIDEO_SOURCE${NC}"
    INPUT_OPTIONS="-i $VIDEO_SOURCE"
fi

echo -e "${GREEN}Multicast address: ${YELLOW}$UDP_HOST:$UDP_PORT${NC}"
echo -e "${GREEN}Bitrate: ${YELLOW}$BITRATE${NC}"
echo ""
echo -e "${YELLOW}Clients can connect using:${NC}"
echo "  ffplay udp://$UDP_HOST:$UDP_PORT"
echo "  vlc udp://@$UDP_HOST:$UDP_PORT"
echo "  mpv udp://$UDP_HOST:$UDP_PORT"
echo ""
echo -e "${GREEN}Starting stream...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Start streaming with infinite loop
# Using multicast for multiple readers
# -stream_loop -1: Loop input file infinitely (for file sources)
# -c:v libx264: H.264 video codec (widely compatible)
# -preset ultrafast: Fast encoding
# -tune zerolatency: Minimize delay
# -f mpegts: MPEG Transport Stream format (good for UDP)

# Add loop option for file sources
if [[ -f "$VIDEO_SOURCE" ]]; then
    LOOP_OPTION="-stream_loop -1"
else
    LOOP_OPTION=""
fi

# Infinite loop to restart stream if it stops
while true; do
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Starting/Restarting stream...${NC}"
    
    ffmpeg $LOOP_OPTION $INPUT_OPTIONS \
        -c:v libx264 \
        -preset ultrafast \
        -tune zerolatency \
        -b:v $BITRATE \
        -f mpegts \
        "udp://$UDP_HOST:$UDP_PORT?pkt_size=1316" 2>&1 | \
        grep -v "deprecated pixel format" | \
        grep -v "frame=" || true
    
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Stream ended, restarting in 2 seconds...${NC}"
    sleep 2
done

echo -e "\n${RED}Stream stopped${NC}"