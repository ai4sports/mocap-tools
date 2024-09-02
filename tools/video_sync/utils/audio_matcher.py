from scipy.io import wavfile
import os
import glob
from .audio_offset_finder import find_offset_between_buffers, InsufficientAudioException, cross_correlation_cuda
from .video_tools import video_cut, get_video_info, video_frame_merge, audio_from_video, get_audio_file
import numpy as np
from multiprocessing.pool import Pool
import argparse
import torchaudio as ta
import torch.nn.functional as f
import torch


AUDIO_MATCH_CFGS = {
    "sample_rate": 16000, # the audio data sampling rate
    "nfft": 1024,
    "bin_stride": 128,  # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
    "bin_length": 256,  # the audio feature bin size
    "n_bins_for_matching": torch.inf,  # number of bins used for matching
    "audio_trim_length": 60 * 20  # the audio will be trimed no longer then this value (seconds)
}



class AudioMatcher:

    def __init__(self, base_audio_file, cfg):
        # self.base_audio, sample_rate = ta.load(base_audio_file, normalize=True)
        self.sample_rate = cfg['sample_rate']
        self.nfft = cfg['nfft']
        self.bin_stride = cfg['bin_stride']
        self.bin_length = cfg['bin_length']
        self.max_frames = cfg['n_bins_for_matching']
        self.audio_trim_length = cfg['audio_trim_length']
        # assert sample_rate == self.sample_rate

        self.base_audio_wave = self.load_audio(base_audio_file)

        self.MFCC = ta.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=26,
                                       melkwargs={"n_fft": self.nfft, "hop_length": self.bin_stride, "n_mels": 26, "center": False}).cuda()
        self.base_audio_mfcc = self.mfcc_normalize(self.MFCC(self.base_audio_wave))

    def cross_correlation_cuda(self, mfcc1, mfcc2, nframes):
        n1, mdim1 = mfcc1.shape
        n2, mdim2 = mfcc2.shape
        o_min = nframes - n2
        o_max = n1 - nframes + 1
        mfcc1_torch = mfcc1.cuda()
        mfcc2_torch = mfcc2.cuda()
        c_right = \
        f.conv1d(mfcc2_torch.transpose(0, 1)[None], mfcc1_torch[:nframes].transpose(0, 1)[:, None], None, stride=1,
                 groups=26)[0].norm(dim=0)
        c_right = c_right[1:].__reversed__()
        c_left = \
        f.conv1d(mfcc1_torch.transpose(0, 1)[None], mfcc2_torch[:nframes].transpose(0, 1)[:, None], None, stride=1,
                 groups=26)[0].norm(dim=0)
        c = torch.cat([c_left, c_right])
        return c, o_min, o_max

    def load_audio(self, audio_f):
        assert audio_f.endswith(".wav")
        audio_wave, sample_rate = ta.load(audio_f, normalize=True)
        assert sample_rate == self.sample_rate
        return audio_wave.cuda()

    def mfcc_normalize(self, mfcc):
        mfcc = mfcc[0].transpose(1,0)
        return (mfcc-mfcc.mean(dim=0))/mfcc.std(dim=0)

    def find_offset(self, ref_audio_f):
        ref_audio_wave = self.load_audio(ref_audio_f)
        ref_audio_mfcc = self.mfcc_normalize(self.MFCC(ref_audio_wave))

        mfcc1 = self.base_audio_mfcc
        mfcc2 = ref_audio_mfcc

        # Derive correl_nframes from the length of audio supplied, to avoid buffer overruns
        correl_nframes = min(len(mfcc1) // 2, len(mfcc2) // 2, self.max_frames)
        if correl_nframes < 10:
            raise InsufficientAudioException(
                "Not enough audio to analyse - try longer clips, less trimming, or higher resolution."
            )

        c, earliest_frame_offset, latest_frame_offset = self.cross_correlation_cuda(mfcc1, mfcc2, nframes=correl_nframes)
        # c = c/np.sqrt(correl_nframes)
        # max_k_index = np.argmax(c)
        max_k_index = torch.argmax(c)
        max_k_frame_offset = max_k_index
        if max_k_index > len(c) / 2:
            max_k_frame_offset -= len(c)
        time_scale = self.bin_stride / self.sample_rate
        time_offset = (max_k_frame_offset) * time_scale

        # std = np.std(c) / np.sqrt(correl_nframes)
        c = f.softmax(c/np.sqrt(correl_nframes))
        # if std < 1e-10:
        #     score = float('inf')
        # else:
        score = c[max_k_index]  # standard score of peak
        return {
            "time_offset": time_offset,
            "frame_offset": int(max_k_index),
            "standard_score": score,
            "correlation": c,
            "time_scale": time_scale,
            "earliest_frame_offset": int(earliest_frame_offset),
            "latest_frame_offset": int(latest_frame_offset),
        }



def find_offset_between_multi_files(base_audio, file2, fs=48000, trim=60 * 20, hop_length=128, win_length=256,max_frames=100000, nfft=512, audio_file=None):
    if audio_file is None or not os.path.exists(audio_file):
        tmp2 = audio_from_video(file2, fs, trim, audio_file)
    else:
        tmp2 =  audio_file
    a2 = wavfile.read(tmp2, mmap=True)[1].astype(float)
    offset_dict = find_offset_between_buffers(base_audio, a2, fs, hop_length, win_length, nfft, max_frames=max_frames)
    if audio_file is None:
        os.remove(tmp2)
    return offset_dict


def get_video_time_offset(video_file_lists,video_names,audio_mathching_cfgs, tmp_audio_root):
    fs = audio_mathching_cfgs['sample_rate']
    trim = audio_mathching_cfgs['audio_trim_length']

    print("Getting time offset by audio matching...")
    base_video = video_file_lists[0]
    base_audio_tmp = get_audio_file(video_names[0], tmp_audio_root)
    base_tmp = audio_from_video(base_video, fs, trim, base_audio_tmp)
    base_audio = wavfile.read(base_tmp, mmap=True)[1].astype(float)
    offset_lists = []
    for i in range(1,len(video_file_lists)):
        audio_tmp = get_audio_file(video_names[i], tmp_audio_root)
        offset = find_offset_between_multi_files(base_audio, video_file_lists[i], fs=fs, trim=trim,
                                                            hop_length=audio_mathching_cfgs["bin_stride"],
                                                            win_length=audio_mathching_cfgs["bin_length"],
                                                            max_frames=audio_mathching_cfgs["n_bins_for_matching"], audio_file=audio_tmp
                                                 )
        print("{} start {}s after {}, matching score: {}".format(video_names[i], offset['time_offset'], video_names[0], offset["standard_score"]))
        offset_lists.append(offset)
    # os.remove(base_tmp)

    offset_times = [0, ]
    for offset in offset_lists:
        offset_times.append(offset["time_offset"])
    offset_times = np.array(offset_times)

    cut_start_times = offset_times.max() - offset_times
    return cut_start_times


def get_video_durations(video_file_lists,video_names, cut_start_times):
    print("Getting trim durations...")

    duration_lists = []
    for video in video_file_lists:
        video_info = get_video_info(video)
        video_stream = next((stream for stream in video_info['streams'] if stream['codec_type'] == 'video'), None)
        duration = float(video_stream['duration'])
        duration_lists.append(duration)

    rest_durations = np.array(duration_lists) - cut_start_times
    video_durations = get_end_time(rest_durations)
    for i ,v in enumerate(video_names):
        print("{} start time: {}  length: {}".format(v, cut_start_times[i], video_durations[i]))
    return video_durations

def cut_multi_videos(video_file_lists, video_names, start_times, video_durations, resolution, output_root,merged_video_path, out_merged_video=False, num_works=16):
    pool = Pool(num_works)
    trimed_videos = []
    for i, video in enumerate(video_file_lists):
        start_time = start_times[i]
        print("Trimming video {}...".format(video))
        output_path = os.path.join(output_root, video_names[i])
        if num_works > 1:
            pool.apply_async(video_cut, (video, start_time, video_durations[i], output_path, resolution))
        else:
            video_cut(video, start_time, video_durations[i], output_path, resolution)
        trimed_videos.append(output_path)
    pool.close()
    pool.join()

    if out_merged_video:
        if resolution == '-1' or int(resolution.split(':')[0]) > 720:
            re_scale = True
        else:
            re_scale = False
        video_frame_merge(trimed_videos,video_names, merged_video_path, re_scale)


def get_end_time(rest_durations):
    sort_index = np.argsort(rest_durations)
    n_total_video = len(rest_durations)
    n_rest_video = n_total_video
    video_durations = np.array(rest_durations)
    for i,idx in enumerate(sort_index):
        n_rest_video -= 1
        if n_rest_video / n_total_video < 0.25:
            for res_idx in sort_index[i:]:
                video_durations[res_idx] = video_durations[idx]
            break
    return video_durations



def align_videos(args, audio_matching_cfgs):
    video_path_root = args.video_path_root
    video_file_lists = glob.glob(os.path.join(video_path_root, args.video_files_parser))
    video_names = [vf[len(video_path_root) + 1:] for vf in video_file_lists]
    out_path = args.out_path
    print("{} videos will be aligned".format(len(video_file_lists)))

    # get offset via audio matching
    cut_start_times = get_video_time_offset(video_file_lists,video_names, audio_matching_cfgs)

    # get trimed duration
    video_durations = get_video_durations(video_file_lists,video_names, cut_start_times)

    resoluation = args.out_resolution

    # cut videos
    cut_multi_videos(video_file_lists,video_names, cut_start_times, video_durations, resoluation, out_path, args.merged_video_path, args.out_merged_video)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find offset between videos by audio matching, then trim to get time-aligned videos "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path_root", metavar="video files root", type=str, help="video files root")
    parser.add_argument("video_files_parser", metavar="video files parser", type=str, help="video files parser to be aligned (relative to video path root), e.g. ./videos/today/sports_*.mp4")
    parser.add_argument("--out-path", default="./tmp", metavar="out_path_root", type=str, help="The root path trimed videos to be saved, the relative paths from root are the same as input videos ")
    parser.add_argument("--out-resolution", type=str, help="The resolution of the trimed video. "
                                                           "e.g. '1080:-1', '720:-1' means the aspect ratio will be keeped, "
                                                           "and the width will be scaled to 1080 or 720. "
                                                           " if this value is '-1' means the original resolution will be used. ",
                        default="720:-1")
    parser.add_argument("--out-merged-video", action='store_true', help='Whether to output a side-by-side merged video of all trimed videos. '
                                                                        'When merging, the trimed video will be scale to 720p')
    parser.add_argument("--merged-video-path", type=str, default="none", help='The path to save the merged video, e.g. output/merged.mp4')


    args = parser.parse_args()
    if args.out_merged_video:
        assert args.merged_video_path != 'none', "The merged video path should be set"

    audio_match_cfgs = {
        "sample_rate":16000, # the audio data sampling rate
        "bin_stride": 128,  # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
        "bin_length": 256, # the audio feature bin size
        "n_bins_for_matching" : 100000, # number of bins used for matching
        "audio_trim_length": 60*20 # the audio will be trimed no longer then this value (seconds)
    }

    align_videos(args, audio_match_cfgs)


if __name__ == '__main__':
    main()