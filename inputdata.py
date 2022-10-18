"""
Defines classes for different sources of input data, to be used with pyramid_main.py
"""

import os
import glob
import sys

import json
import numpy as np
import imageio
from skimage.transform import resize

import skvideo
FFMPEGPATH = "/cluster/research-groups/wehrwein/ffmpeg-share"
skvideo.setFFmpegPath(FFMPEGPATH)
import skvideo.io

SECS_PER_HOUR = 3600

with open("datasets.json", 'r') as f:
    ALL_DATA_DICT = json.load(f)


def make_resultpath(pathname):
    try:
        os.makedirs(pathname)
        return pathname
    except FileExistsError:
        return pathname


def resize_if_too_big(img_array):
    while (img_array.shape[0] * img_array.shape[1]) > 6250000:
        h = img_array.shape[0] / 2
        w = img_array.shape[1] / 2
        c = img_array.shape[2]
        img_array = resize(img_array, (h,w,c), anti_aliasing=True)
        if img_array.dtype != 'uint8':
            img_array = (img_array*255).astype('uint8')
    return img_array


class InputData():

    def __init__(self, dataset_name):
        datadict = ALL_DATA_DICT[dataset_name]
        self.name = datadict["filepath_name"]
        self.dirpath = datadict["dirpath"]
        self.form = datadict["form"]
        self.baselvl = datadict["baselvl"]
        self.fps = datadict["fps"]
        self.vidhrs = datadict["vid_hrs_1day"]
        self.fpv = datadict["frames_per_vid"]
        self.pyr_result_path = make_resultpath(datadict["pyr_result_path"])

    def get_multi_blur_globstring_all_days(self, level):
        return self.pyr_result_path + "/*" + self.name + "_" + str(level).zfill(2) + "_blur.mp4"

    def get_multi_diff_globstring_all_days(self, level):
        return self.pyr_result_path + "/*" + self.name + "_" + str(level).zfill(2) + "_diff.mp4"

    def get_multi_blur_globstring_one_year(self, level, year):
        return self.pyr_result_path + "/" + str(year)[-2:] + "*" + self.name + "_" +\
                str(level).zfill(2) + "_blur.mp4"

    def get_multi_diff_globstring_one_year(self, level, year):
        return self.pyr_result_path + "/" + str(year)[-2:] + "*" + self.name + "_" +\
                str(level).zfill(2) + "_diff.mp4"

    def get_multi_blur_globstring_years(self, level):
        return self.pyr_result_path + "/" + self.name + "_" + str(level).zfill(2) + "_blur*.mp4"

    def get_multi_diff_globstring_years(self, level):
        return self.pyr_result_path + "/" + self.name + "_" + str(level).zfill(2) + "_diff*.mp4"

    def get_multi_blur_globstring(self, level, year=None):
        if level < 16 and year is None:
            return self.get_multi_blur_globstring_all_days(level)
        if level < 16 and year is not None:
            return self.get_multi_blur_globstring_one_year(level, year)
        return self.get_multi_blur_globstring_years(level)

    def get_multi_diff_globstring(self, level, year=None):
        if level < 16 and year is None:
            return self.get_multi_diff_globstring_all_days(level)
        if level < 16 and year is not None:
            return self.get_multi_diff_globstring_one_year(level, year)
        return self.get_multi_diff_globstring_years(level)

    def get_blur_path_one_year(self, lv_num, year):
        level = "_" + str(lv_num).zfill(2)
        return self.pyr_result_path + "/" + self.name + level + "_blur_" + str(year) + ".mp4"

    def get_diff_path_one_year(self, lv_num, year):
        level = "_" + str(lv_num).zfill(2)
        return self.pyr_result_path + "/" + self.name + level + "_diff_" + str(year) + ".mp4"

    def get_blur_path(self, lv_num, year=None):
        level = "_" + str(lv_num).zfill(2)
        if year is None:
            return self.pyr_result_path + "/" + self.name + level + "_blur.mp4"
        return self.get_blur_path_one_year(lv_num, year)

    def get_diff_path(self, lv_num, year=None, sub_lvl=-1):
        level = "_" + str(lv_num).zfill(2)
        if year is None:
            return self.pyr_result_path + "/" + self.name + level + "_diff.mp4"
        return self.get_diff_path_one_year(lv_num, year)

    def get_black_frame_save_path(self):
        return self.pyr_result_path + "/black_frame_" + self.name + ".npy"

    def get_black_frame_img(self):
        black_frame_path = self.get_black_frame_save_path()
        try:
            img = np.load(black_frame_path)
            img = img.astype('float32')
            return img
        except FileNotFoundError:
            print("Black frame file not found, cannot proceed.")
            sys.exit()

    def frame_gen(self, level, year=None):
        """
        Generator that yields images (as ndarrays) one at a time,
        from a blur video at this level.
        """
        vidpath = self.get_blur_path(level, year)
        print("getting frames from:", vidpath)
        if os.path.exists(vidpath):
            vreader = skvideo.io.vreader(vidpath)
            for frame in vreader:
                yield frame
        else:
            print("input video path is not valid")
            print("path =", vidpath)

        vreader.close()

    def get_base_level(self):
        """ This method returns the desired base level, fps and sampling rate
        for the dataset. The default behavior is to eliminate construction of levels 1-3
        due to the large size of 30fps videos and the time it takes to process;
        override this by putting the string "full" in the filepath_name
        in datasets.json.
        """
        if self.baselvl == 0 and "full" not in self.name:
            baselvl = 3
            fps = 1
            sampl_rate = self.fps
        else:
            baselvl = self.baselvl
            fps = self.fps
            sampl_rate = 1
        return baselvl, fps, sampl_rate

    def get_stride(self, level):
        """
        Level 0: 1 frame per 0.0333 secs (30 fps)
        Level 1: 1 frame per 0.1667 secs (6 fps): 5x
        Level 2: 1 frame per 0.5 secs (2 fps): 3x
        Level 3: 1 frame per 1 sec (1 fps): 2x
        Level 4: 1 frame per 5 secs (0.2 fps): 5x
        Level 5: 1 frame per 15 secs: 3x
        Level 6: 1 frame per 30 secs: 2x
        Level 7: 1 frame per 60 secs / 1 min: 2x
        Level 8: 1 frame per 5 mins: 5x
        Level 9: 1 frame per 15 mins: 3x
        Level 10: 1 frame per 30 mins: 2x
        Level 11: 1 frame per 60 mins / 1 hr: 2x
        Level 12: 1 frame per 2 hrs: 2x
        Level 13: 1 frame per 4 hrs: 2x
        Level 14: 1 frame per 12 hrs (varies, if vids span < 24 hrs): 3x or other
        Level 15: 1 frame per 24 hrs / 1 day: 2x
        Level 16: 1 frame per 3 days: 3x
        Level 17: 1 frame per 6 days: 2x
        Level 18: 1 framer per 30 days: 5x
        Level 19: 1 frame per 90 days: 3x
        Level 20: 1 frame per 180 days: 2x
        Level 21: 1 frame per 360 days (approx 1 frame per year): 2x
        Levels 22+ will go up by powers of 2 (1 frame per 2 'years', per 4 'years', etc)
        """
        base, _f, _s = self.get_base_level()

        master_stride_list = [5,3,2,5,3,2,2,5,3,2,2,2,2,3,2,3,2,5,3,2,2]
        stride_list = [0]*base + master_stride_list[base:]

        if self.vidhrs == 16:
            stride_list[13] = 2
        elif self.vidhrs == 12:
            stride_list[14] = -1
        elif self.vidhrs == 8:
            stride_list[13] = -1

        if level > 21:
            return 2
        return stride_list[level - 1]

    def get_blur_filter(self, level=15):
        """
        Get blur filter appropriate for the stride for level.
        """
        stride = self.get_stride(level)
        if stride == 2:
            return (1,2,2,1)
        if stride == 3:
            return (1,2,3,2,1)
        if stride == 5:
            return (1,2,3,4,5,4,3,2,1)
        return (None,)

    def get_reconstruct_path(self, recon_level, top_level, subtract, part=-1):
        base_path = self.pyr_result_path + "/reconstructions"
        if part == -1:
            part_str = ""
        else:
            part_str = "part-" + str(part).zfill(2)

        if len(subtract) == 0:
            subtract_str = ""
        else:
            subtract_str = "_exclude_"
            for num in subtract:
                subtract_str += str(num).zfill(2) + "_"

        filename = self.name + "_reconst-to-" + str(recon_level) + "-from-" +\
                str(top_level) + subtract_str + part_str + ".mp4"

        return os.path.join(base_path, filename)

    def get_diff_frames(self, lvl_num, weight):
        """
        Generate normalized, weighted frames from the
        diff file for a level.
        """
        #TODO allow for multi-day diff files that are too long and in multiple parts
        diff_path = self.get_diff_path(lvl_num)
        if os.path.exists(diff_path):
            vreader = skvideo.io.vreader(diff_path)
            for frame in vreader:
                frame = (frame.astype('float32') / 255) - 0.5
                weighted_frame = frame * weight
                yield weighted_frame
        else:
            print("file does not exist:", diff_path)
            sys.exit()

class InputDataOneDay(InputData):

    def __init__(self, dataset_name, yymmdd):

        super().__init__(dataset_name)
        self.date = yymmdd

        # the conditional statements below may need to be updated or added to if you
        # use your own custom dataset, depending on your file naming convention and
        # where the year, month, and day are designated in the filenames.
        if self.name == "buxtoncoastalcam":
            yr = "20" + yymmdd[:2]
            mo = yymmdd[2:4]
            day = yymmdd[-2:]
            self.globstring = self.dirpath + "/" + self.name + "." + yr + "-"+ mo + "-" + day + "*"
        elif self.name == "hiuchi" or self.name == "kutcharoko":
            self.globstring = self.dirpath + "/20" + yymmdd[:2] + "/" + yymmdd[2:4]\
                    + "/" + yymmdd + "*"
        elif self.name == "smokylook":
            yr = "20" + yymmdd[:2]
            mo = yymmdd[2:4]
            day = yymmdd[-2:]
            self.globstring = self.dirpath + "/" + yr + "/" + mo + "/" + self.name + "_"\
                    + yr + "_" + mo + "_" + day + "*"
        else:
            self.globstring = self.dirpath + "/" + yymmdd + "*"

        self.black_frame = self.get_black_frame_img()

    def get_fps(self):
        return self.fps

    def get_result_path(self):
        return self.pyr_result_path

    def get_blur_path(self, lv_num, year=None):
        """ precondition: lv_num > 0 """
        level = "_" + str(lv_num).zfill(2)
        return self.pyr_result_path + "/" + self.date + self.name + level + "_blur.mp4"

    def get_diff_path(self, lv_num, year=None, sub_lvl=-1):
        """ precondition: lv_num > 0 """
        level = "_" + str(lv_num).zfill(2)
        if sub_lvl != -1:
            sub_lvl = "-" + str(sub_lvl).zfill(2)
        else:
            sub_lvl = ""
        return self.pyr_result_path + "/" + self.date + self.name\
                + level + "_diff" + sub_lvl + ".mp4"

    def get_diff_globstring(self, lv_num):
        level = "_" + str(lv_num).zfill(2)
        return self.pyr_result_path + "/" + self.date + self.name + level + "_diff*.mp4"

    def get_globstring(self):
        return self.globstring

    def get_name(self):
        return self.name

    def get_date(self):
        return self.date

    def get_form(self):
        return self.form

    def get_black_frame_img(self):
        black_frame_path = self.get_black_frame_save_path()
        if os.path.exists(black_frame_path):
            img = np.load(black_frame_path)
        else:
            file_list = glob.glob(self.globstring)
            file_list = [filename for filename in file_list if filename[-4:]!=".txt"]
            filepath = file_list[0]
            if self.form == "images":
                img = imageio.imread(filepath)
                img = resize_if_too_big(img)
                img = np.zeros_like(img)
            elif self.form == "multi-video":
                reader = skvideo.io.vreader(filepath, num_frames=1)
                for frame in reader:
                    img = frame
                    img = resize_if_too_big(img)
                    np.save(black_frame_path, np.zeros_like(img))

        img = img.astype('float32')
        return img

    def get_shape(self):
        template = self.get_black_frame_img()
        return template.shape

    def get_stride_sizes(self):
        sizes = []
        for i in range(1, 16):
            sizes.append(self.get_stride(i))
        return tuple(sizes)

    def get_diff_frames(self, lvl_num, weight):
        """
        Generate normalized, weighted frames from the
        diff file for a level.
        """
        diff_list = glob.glob(self.get_diff_globstring(lvl_num))
        diff_list.sort()
        for filename in diff_list:
            vreader = skvideo.io.vreader(filename)
            try:
                for frame in vreader:
                    frame = (frame.astype('float32') / 255) - 0.5
                    weighted_frame = frame * weight
                    yield weighted_frame
            except:
                print("ERROR reading frame from", filename)
                continue

    def frame_gen(self, level, year=None):
        """
        Generator that yields images (as ndarrays) one at a time,
        either from a collection of images, or from video frames.
        """
        base, fps, sampling_rate = self.get_base_level()

        black_frame = self.get_black_frame_img()
        black_frame = black_frame.astype('uint8')
        total_frames = fps * SECS_PER_HOUR * self.vidhrs
        frame_limit = self.fpv // sampling_rate
        num_vids = total_frames // frame_limit

        if level == base and self.form == "images":
            img_list = glob.glob(self.globstring)
            img_list.sort()
            for filename in img_list:
                print(filename)
                if "MISSING.txt" in filename:
                    yield black_frame
                else:
                    img = imageio.imread(filename)
                    img = resize_if_too_big(img)
                    yield img
        elif level == base and self.form == "multi-video":
            vid_list = glob.glob(self.globstring)
            vid_list.sort()
            vid_count = 0

            for vid in vid_list:
                frame_count = 0
                yield_count = 0
                print("getting frames from:", vid)
                vid_count += 1

                if "MISSING.txt" in vid:
                    while yield_count < frame_limit:
                        yield_count += 1
                        yield black_frame
                else:
                    vreader = skvideo.io.vreader(vid)
                    try:
                        for frame in vreader:
                            if frame_count % sampling_rate == 0:
                                yield_count += 1
                                if yield_count <= frame_limit:
                                    yield frame
                                else:
                                    break
                            frame_count += 1
                        while yield_count < frame_limit:
                            yield_count += 1
                            yield black_frame
                    except:
                        print("ERROR reading frame")
                        print("filling remainder of vid with black frames")
                        while yield_count < frame_limit:
                            yield_count += 1
                            yield black_frame
                        continue

            while vid_count < num_vids:
                yield_count = 0
                while yield_count < frame_limit:
                    yield_count += 1
                    yield black_frame
                vid_count += 1

        else:
            path = self.get_blur_path(level, year)
            print("getting frames from:", path)
            if os.path.exists(path):
                vreader = skvideo.io.vreader(path)
                try:
                    for frame in vreader:
                        yield frame
                except:
                    print("ERROR reading frame from", path)
            else:
                print("input video path is not valid")
                print("path =", path)
