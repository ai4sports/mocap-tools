import os
import warnings
from typing import List, Dict

import numpy as np
import torch

from utils.audio_matcher import AudioMatcher, audio_from_video


class VideoSync:

    def __init__(self, audio_match_cfgs: Dict[str, int] = None):
        """
        Args:
            audio_match_cfgs: Dict[str, int], the audio matching configs, default is
            {
                "sample_rate": 48000,  # the audio data sampling rate
                "nfft": 1024,
                "bin_stride": 128,  # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
                "bin_length": 256,  # the audio feature bin size
                "n_bins_for_matching": torch.inf,  # number of bins used for matching
                "audio_trim_length": 60* 10 # the audio will be trimed no longer then this value (seconds)
            }
        """
        if audio_match_cfgs is None:
            audio_match_cfgs = {
                "sample_rate": 44100,  # the audio data sampling rate
                "nfft": 1024,
                "bin_stride": 128,
                # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
                "bin_length": 256,  # the audio feature bin size
                "n_bins_for_matching": torch.inf,  # number of bins used for matching
                "audio_trim_length": 60 * 10  # the audio will be trimed no longer then this value (seconds)
            }

        self.AUDIO_MATCH_CFGS = audio_match_cfgs

    def time_offset_to_frame_offset(self, time_offset, video_fps):
        frame_offset = time_offset * video_fps
        frame_offset = np.round(frame_offset).astype(int)
        return frame_offset

    def get_video_time_offset(self, videos, camera_names, audio_match_cfgs, audio_path):
        audios = []
        offsets = [0, ]
        # videos = list(videos_dict.values())
        for i, video in enumerate(videos):
            audio_name = os.path.basename(video).split('.')[0] + '.wav'
            audio_file = os.path.join(audio_path, audio_name)
            # os.system('ffmpeg -i {} -vn -acodec copy {}'.format(video, audio_file))
            if os.path.exists(audio_file):
                audios.append(audio_file)
                continue
            audio_from_video(video, audio_match_cfgs['sample_rate'], audio_match_cfgs['audio_trim_length'], audio_file)
            # videos[i] = audio_path
            audios.append(audio_file)
        matcher = AudioMatcher(audios[0], audio_match_cfgs)
        for i in range(1, len(audios)):
            match_res = matcher.find_offset(audios[i])
            time_offset = match_res['time_offset'].item()
            stand_score = match_res["standard_score"].item()
            if stand_score < 0.01:
                warnings.warn("video {} and video {} not match".format(videos[0], videos[i]))
                # raise ValueError("video {} and video {} not match".format(videos[0], videos[i]))
            print(camera_names[i], time_offset)
            offsets.append(time_offset)

        offsets = np.array(offsets)
        ss_time = offsets.max() - offsets
        return ss_time

    def process(self, video_dict: Dict[str, str], video_fps: int = 25):
        """
        Return the frame offset and time offset for each view
        Args:
            video_dict: Dict[str, str], dict of view_id and video path
            video_fps: int, fps of the video
        Returns:
            frame_offset: List[int], list of frame offsets for each view
            time_offset: List[float], list of time offsets for each view
        """
        view_ids = list(video_dict.keys())
        video_list = list(video_dict.values())

        video_dir = os.path.dirname(video_list[0])
        tmp_audio_dir = video_dir
        video_names = [os.path.basename(video) for video in video_list]
        time_offset = self.get_video_time_offset(video_list, video_names, self.AUDIO_MATCH_CFGS, tmp_audio_dir)
        for i, view_id in enumerate(view_ids):
            print(f"View {view_id}: {time_offset[i]}")

        frame_offset = self.time_offset_to_frame_offset(time_offset, video_fps)
        return frame_offset, time_offset


if __name__ == "__main__":
    video_sync = VideoSync()
    video_dict = {
        0: "workdir_all/0616/videos/record_06-16_105558/13.mp4",
        1: "workdir_all/0616/videos/record_06-16_105558/14.mp4",
        2: "workdir_all/0616/videos/record_06-16_105558/15.mp4",
        3: "workdir_all/0616/videos/record_06-16_105558/16.mp4",
        4: "workdir_all/0616/videos/record_06-16_105558/17.mp4"
    }
    video_sync.process(video_dict)
