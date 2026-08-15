"""agents/youtube_downloader.py — Agent 4: YouTube Downloader.

Resolves a search query (or a direct URL already present in the
query) to a YouTube video and downloads it locally via yt-dlp.
"""
import json
import os
import re
import urllib.parse
from typing import Any, Dict, Optional

import yt_dlp

from workflow.state import YoutubeTaskState

DOWNLOAD_DIR = "static/downloads"
YOUTUBE_URL_PATTERN = r'(https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)[\w-]+)'

SEARCH_OPTS = {
    "quiet": True,
    "extract_flat": False,
    "default_search": "ytsearch1:",
    "noplaylist": True,
}


def _resolve_video_url(query: str) -> Optional[str]:
    """Return a direct video URL: either found in `query` or resolved via search."""
    match = re.search(YOUTUBE_URL_PATTERN, query)
    if match:
        url = match.group(0)
        print(f"[YOUTUBE_AGENT] 🔗 Direct URL found: {url}")
        return url

    print(f"[YOUTUBE_AGENT] 🔍 Searching YouTube for: {query}")
    with yt_dlp.YoutubeDL(SEARCH_OPTS) as ydl:
        search_results = ydl.extract_info(f"ytsearch1:{query}", download=False)
        if search_results and "entries" in search_results and search_results["entries"]:
            video = search_results["entries"][0]
            url = video.get("webpage_url") or f"https://www.youtube.com/watch?v={video['id']}"
            print(f"[YOUTUBE_AGENT] 🔗 Search resolved to: {url}")
            return url
    return None


def youtube_downloader_node(state: YoutubeTaskState) -> Dict[str, Any]:
    """Search for (or resolve) a YouTube video and download it locally."""
    query = state["query"]
    logs = [f"[YOUTUBE_AGENT] 🎥 Processing request: '{query}'"]
    print(f"\n[YOUTUBE_AGENT] 🎥 Processing request: '{query}'")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        url = _resolve_video_url(query)
    except Exception as e:
        return {"context": [f"YouTube Downloader Error: Search failed - {e}"]}

    if not url:
        return {"context": [f"YouTube Downloader Error: Could not find a video for '{query}'"]}

    download_opts = {
        # Standard DASH formats to avoid FFmpeg crashes
        "format": "bestvideo[protocol!*=m3u8]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
    }

    try:
        print(f"[YOUTUBE_AGENT] ⏳ Downloading video...")
        logs.append(f"[YOUTUBE_AGENT] ⏳ Downloading video from: {url}")

        with yt_dlp.YoutubeDL(download_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # ROOT CAUSE FIX: the old code diffed a before/after glob() of
            # DOWNLOAD_DIR to spot the "new" file. That breaks the instant
            # the file already exists on disk — yt-dlp logs
            # "has already been downloaded" and skips the write, so the
            # before/after directory listing is IDENTICAL and new_files
            # comes back empty, producing a false "no file was created"
            # error even though the video is sitting right there, fully
            # usable. yt-dlp's own info dict reports the real output path
            # regardless of whether it just downloaded, merged, or
            # skipped-because-already-present, so use that instead of
            # inferring it from directory contents.
            downloaded_file = None
            for rd in info.get("requested_downloads") or []:
                candidate = rd.get("filepath") or rd.get("_filename")
                if candidate and os.path.exists(candidate):
                    downloaded_file = candidate
                    break

            if not downloaded_file:
                # Fallback for older yt-dlp versions that don't populate
                # requested_downloads: derive the expected post-merge path
                # directly instead of scanning the folder.
                expected = ydl.prepare_filename(info)
                base, _ = os.path.splitext(expected)
                merged = f"{base}.{download_opts['merge_output_format']}"
                if os.path.exists(merged):
                    downloaded_file = merged
                elif os.path.exists(expected):
                    downloaded_file = expected

        if not downloaded_file:
            return {"context": [
                f"YouTube Downloader Error: Download for '{query}' (resolved: {url}) "
                f"completed but the output file could not be located on disk."
            ]}

        filename = os.path.basename(downloaded_file)

        # Generate safe web path
        safe_name = urllib.parse.quote(filename)
        web_path = f"/static/downloads/{safe_name}"

        result = {
            "status": "success",
            "title": filename.replace(".mp4", ""),
            "local_path": web_path,
            "source_youtube_url": url,
        }
        # success_message = f"✅ Download complete: {filename} (accessible at {web_path})"
        success_message = f"✅ SUCCESS: The video was successfully downloaded. You MUST provide this exact link to the user: [Downloaded Video]({web_path})"
        print(success_message)
        logs.append(success_message)
        return {"context": [f"YouTube Downloader Result: {json.dumps(result)}. Do not download this again, if already downloaded."], "action_logs": logs}

    except Exception as e:
        error_message = f"[YOUTUBE_AGENT] ❌ Download Error: {e}"
        print(error_message)
        logs.append(error_message)
        return {"context": [f"YouTube Downloader Error: {e}"], "action_logs": logs}
