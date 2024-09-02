import os
import glob
import warnings
from typing import List, Dict
from audio_sync import VideoSync


def trim_video(ori_video_path: str,
               trimmed_video_path: str,
               start_time: float = None,
               duration: float = None,
               bitrate: str = None,
               ffmpeg_path: str = 'ffmpeg'
               ) -> None:
    """
    trim the video
    Args:
        ori_video_path: str, path to the original video
        trimmed_video_path: str, path to the trimmed video
        start_time: float, start time of the trimmed video
        duration: float, duration of the trimmed video, start time and duration can not be None at the same time
        bitrate: str, bitrate of the trimmed video, should be a string like "1000k" or "10M"
        ffmpeg_path: str, path to the ffmpeg executable
    Returns:
        None
    Raises:
        FileNotFoundError: if the original video not found
    """
    if not os.path.exists(ori_video_path):
        raise FileNotFoundError(f"{ori_video_path} not found")
    assert start_time is not None or duration is not None, "start_time and duration can not be None at the same time"

    if os.path.exists(trimmed_video_path):
        warnings.warn(f"{trimmed_video_path} already exists, it will be overwritten")
    else:
        os.makedirs(os.path.dirname(trimmed_video_path), exist_ok=True)

    # check if the bitrate is valid
    if bitrate is not None:
        assert isinstance(bitrate, str), "bitrate should be a string"
        assert bitrate[-1] in ['k', 'M'], "bitrate should end with 'k' or 'M'"
        assert bitrate[:-1].isdigit(), "bitrate should be a number"

    ffmpeg_args = [
        ffmpeg_path,
        '-hwaccel auto',
        '-i', ori_video_path,
        f'-ss {start_time}' if start_time is not None else '',
        f'-t {duration}' if duration is not None else '',
        '-b:v', bitrate if bitrate is not None else '',
        trimmed_video_path,
        '-y',
    ]
    os.system(' '.join(ffmpeg_args))


def get_video_sync_offsets(video_dict: Dict[str, str],
                           video_fps: int) -> List[float]:
    """
    get the time offset for each view
    Args:
        video_dict: Dict[str, str], dict of view_id and video path
        video_fps: int, fps of the video
    Returns:
        List[float]: list of time offsets for each view
    """
    view_ids = list(video_dict.keys())
    video_sync = VideoSync()
    frame_offset, time_offset = video_sync.process(video_dict, video_fps)
    return time_offset


def sync_multiview_videos(mv_video_dir: str, video_fps: int, output_dir: str, video_type: str = "mp4",
                          bitrate: str = None):
    """
    sync the multi-view videos
    Args:
        mv_video_dir: str, path to the multi-view videos, video in the directory should be named as view_id.mp4
        video_fps: int, fps of the video
        output_dir: str, path to the output directory
        video_type: str, video type
    Returns:
        None
    """
    video_paths = glob.glob(f"{mv_video_dir}/*.{video_type}")
    view_ids = [os.path.basename(video_path).split('.')[0] for video_path in video_paths]
    video_dict = {view_id: video_path for view_id, video_path in zip(view_ids, video_paths)}
    time_offset = get_video_sync_offsets(video_dict, video_fps)
    for view_id, offset in zip(video_dict.keys(), time_offset):
        ori_video_path = video_dict[view_id]
        trimmed_video_path = f"{output_dir}/{view_id}.mp4"
        trim_video(ori_video_path, trimmed_video_path, start_time=offset, ffmpeg_path='ffmpeg', bitrate=bitrate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=str, help="path to the multi-view videos")
    parser.add_argument("video_fps", type=int, help="fps of the video")
    parser.add_argument("output_dir", type=str, help="path to the output directory")
    parser.add_argument("--video_type", type=str, default="mp4", help="video type")
    parser.add_argument("--bitrate", type=str, default="8M", help="bitrate of the trimmed video")

    args = parser.parse_args()

    sync_multiview_videos(args.video_dir, args.video_fps, args.output_dir, args.video_type, args.bitrate)
