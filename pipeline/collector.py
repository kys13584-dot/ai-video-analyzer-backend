import yt_dlp # type: ignore
import os
import uuid
import glob
from typing import Dict, Any, Optional, Tuple

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

SUPPORTED_DOMAINS = [
    "youtube.com", "youtu.be",          # YouTube (including Shorts)
    "instagram.com",                    # Instagram Reels / Posts
    "tiktok.com",                       # TikTok
    "twitter.com", "x.com",             # Twitter/X
    "facebook.com",                     # Facebook Reels / Videos
]


def is_supported_url(url: str) -> bool:
    """Returns True if the URL is from a known supported platform."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in SUPPORTED_DOMAINS)


class VideoCollector:
    def __init__(self, download_dir: str = DOWNLOAD_DIR):
        self.download_dir = download_dir

    def download_video(self, url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Downloads a video from a URL using yt-dlp.
        Supports YouTube (Shorts), Instagram Reels, TikTok, Twitter/X, and Facebook.

        Returns:
            (metadata_dict, None) on success
            (None, error_message) on failure
        """
        video_id = str(uuid.uuid4())
        output_template = os.path.join(self.download_dir, f"{video_id}.%(ext)s")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Instagram / TikTok often require a fake user agent
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/122.0.0.0 Safari/537.36'
                ),
            },
            # Retry logic for network hiccups
            'retries': 3,
            'fragment_retries': 3,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)

                # Handles playlists — just take the first entry
                if 'entries' in info_dict:
                    info_dict = info_dict['entries'][0]

                filepath = ydl.prepare_filename(info_dict)

                # yt-dlp가 실제로 생성한 파일명이 prepare_filename과 다를 수 있음
                # (포맷 코드 포함 예: {uuid}.f399.mp4, 또는 .m4a 등)
                # → UUID 프리픽스로 glob 탐색해 실제 파일을 찾음
                if not os.path.exists(filepath):
                    matches = [
                        f for f in glob.glob(os.path.join(self.download_dir, f"{video_id}*"))
                        if not f.endswith('.part') and os.path.isfile(f)
                    ]
                    if matches:
                        filepath = matches[0]

                metadata = {
                    "source_url": url,
                    "title": info_dict.get('title', 'Unknown Title'),
                    "duration": info_dict.get('duration', 0.0),
                    "resolution": f"{info_dict.get('width', 0)}x{info_dict.get('height', 0)}",
                    "file_path": filepath,
                }

                return metadata, None

        except Exception as e:
            error_msg = str(e)
            print(f"[Collector] Error downloading {url}: {error_msg}")
            return None, error_msg

