"""
Defines methods used for building video temporal pyramids.
"""

import os
import shutil
import glob
from queue import Queue

import numpy as np
import skvideo.io


def set_up_writer(outpath):
    """
    Creates and returns an skvideo FFmpegWriter object.
    Requires import of skvideo.io module.
    """
    fps_30_input = {'-framerate':'30/1'}
    fps_30_output = {'-r':'30/1'}
    return skvideo.io.FFmpegWriter(
            outpath,
            inputdict=fps_30_input,
            outputdict=fps_30_output
    )


def create_diff_img(orig, avg):
    """
    Returns a 'diff' image that is created by subtracting avg from orig,
    then clipping to values between -0.5 and 0.5, then adding 0.5 so that
    the pixel values are once again between 0 and 1.
    Precondition: images are normalized (float32, values between 0 and 1)
    """
    if np.sum(orig) == 0:
        return np.full(orig.shape, 0.5)

    diff = orig - avg
    diff = np.clip(diff, -0.5, 0.5)
    diff += 0.5
    return diff


def read_original_frames(input_obj, level, padding_size, orig_queue, avg_coroutine,\
        year=None):
    """
    Reads original frames, adds front and back padding if needed, and
    sends out the frames to a coroutine to be filtered/processed.
    Padded frames are divided by 2 to give them less weight than real
    frames.  Output frames are normalized, float values.
    Original frames are also saved out to origQueue for use in creating diffs.
    """
    front_pad_count = padding_size
    last_frame = []
    base, _f, _s = input_obj.get_base_level()
    print("level:", level, "year:", year, "base:", base)
    frames = input_obj.frame_gen(level, year)

    for frm in frames:
        frm = frm.astype('float32') / 255
        orig_queue.put(frm)

        if len(last_frame) == 0:
            last_frame.append(frm)
        else:
            last_frame[0] = frm

        while front_pad_count > 0:
            front_pad = frm / 2
            avg_coroutine.send(front_pad)
            front_pad_count -= 1

        avg_coroutine.send(frm)

    last_pad = last_frame[0] / 2
    for j in range(padding_size):
        avg_coroutine.send(last_pad)


def create_filtered_avg_frames(blur_filter, stride, file_coroutine, upsample_coroutine, zeroframe):
    """
    Coroutine that consumes frames (normalized) and then sends averaged frames to
    the next coroutines (one to write the blur file, and one to upsample for
    creating the diff file).
    """
    frame_count = 0
    filter_size = len(blur_filter)

    #two queues will alternate providing the average frame
    q_blur_1 = Queue(maxsize = filter_size)
    q_blur_2 = Queue(maxsize = filter_size)
    q_blur_2_active = False
    skip_frame_1 = False
    skip_frame_2 = False
    non_zero_count_1 = 0
    non_zero_count_2 = 0
    avg_count = 0

    #initialize index into blur_filter tuple, for weights
    idx1 = 0
    idx2 = 0

    try:
        while True:
            frm = (yield)

            if frame_count == 0:
                total1 = np.zeros_like(zeroframe)
                total2 = np.zeros_like(zeroframe)

            frame_count += 1
            if frame_count == stride + 1:
                q_blur_2_active = True

            wt1 = blur_filter[idx1]

            if skip_frame_1:
                skip_frame_1 = False
            else:
                if not q_blur_1.full() and not skip_frame_1:
                    q_blur_1.put(frm * wt1)
                    total1 += (frm * wt1)
                    if np.sum(frm) != 0:
                        non_zero_count_1 += wt1
                if q_blur_1.full():
                    if non_zero_count_1 != 0:
                        avg1 = total1 / non_zero_count_1
                    else:
                        avg1 = total1

                    file_coroutine.send(avg1)
                    upsample_coroutine.send(avg1)

                    avg_count += 1

                    while not q_blur_1.empty():
                        q_blur_1.get()
                    total1 = np.zeros_like(zeroframe)
                    non_zero_count_1 = 0
                    if stride % 2 == 1:
                        skip_frame_1 = True

                idx1 += 1
                if idx1 == filter_size:
                    idx1 = 0

            if skip_frame_2:
                skip_frame_2 = False
            elif q_blur_2_active and not skip_frame_2:
                wt2 = blur_filter[idx2]
                if not q_blur_2.full():
                    q_blur_2.put(frm * wt2)
                    total2 += (frm * wt2)
                    if np.sum(frm) != 0:
                        non_zero_count_2 += wt2
                if q_blur_2.full():
                    if non_zero_count_2 != 0:
                        avg2 = total2 / non_zero_count_2
                    else:
                        avg2 = total2

                    file_coroutine.send(avg2)
                    upsample_coroutine.send(avg2)

                    avg_count += 1

                    while not q_blur_2.empty():
                        q_blur_2.get()
                    total2 = np.zeros_like(zeroframe)
                    non_zero_count_2 = 0

                    if stride % 2 == 1:
                        skip_frame_2 = True

                idx2 += 1
                if idx2 == filter_size:
                    idx2 = 0

    except GeneratorExit:
        print("Done with filtering frames!")


def write_to_file(writer, curr_path="filepath.mp4"):
    """
    Coroutine that consumes normalized image frames, and
    changes them back to uint8 type, and then
    writes them to a video file using an FFmpegWriter object.
    The currPath variable should be populated for real when it is likely
    that the writer will end up writing more than 432,000
    frames (i.e., a 4 hour video).
    """
    blur_in_path = ("blur" in curr_path)
    part = 2
    curr_path_base = curr_path[:-4]
    curr_path_ext = curr_path[-4:]  # assumes .mp4 or similar 3 letter extension
    new_path = curr_path_base + "-" + str(part).zfill(2) + curr_path_ext
    frame_count = 0
    try:
        while True:
            frm = (yield)
            fr_real = (frm * 255).astype('uint8')
            frame_count += 1
            if frame_count % 100 == 0:
                print("curr_path:", curr_path, "- writing frame:", frame_count)
            writer.writeFrame(fr_real)

            if frame_count % 432000 == 0 and not blur_in_path:
                writer.close()
                writer = set_up_writer(new_path)
                part += 1
                new_path = curr_path_base + "-" + str(part).zfill(2) + curr_path_ext

    except GeneratorExit:
        writer.close()
        part2_path = curr_path_base + "-" + "02" + curr_path_ext
        part1_path = curr_path_base + "-" + "01" + curr_path_ext
        if os.path.exists(part2_path):
            os.rename(curr_path, part1_path)
        print("Done with writing file!")


def upsample_and_pad(stride, filt, filter_coroutine):
    """
    Coroutine that consumes normalized image arrays which are avg/blur
    frames, and upsamples based on stride, creating padding based
    on the size of the filter this time (not the size of the stride),
    then sends these frames on to the next coroutine to get
    blurred/filtered.
    """
    filter_size = len(filt)
    padding_size = filter_size // 2
    front_pad_count = padding_size
    try:
        last_frame = []
        while True:
            frm = (yield)
            if frm.dtype == 'uint8':
                frm = frm.astype('float32') / 255

            if len(last_frame) == 0:
                last_frame.append(frm)
            else:
                last_frame[0] = frm

            while front_pad_count > 0:
                front_pad = frm / 2
                filter_coroutine.send(front_pad)
                front_pad_count -= 1

            for i in range(stride):
                filter_coroutine.send(frm)

    except GeneratorExit:
        last_pad = last_frame[0] / 2
        if filter_size % 2 == 0:
            padding_size -= 1
        for j in range(padding_size):
            filter_coroutine.send(last_pad)
        print("Done with upsample & pad!")


def filter_stride_1(filt, diff_coroutine, zeroframe):
    """
    Coroutine that consumes upsampled frames (including padding)
    and filters them with same filter that created them in
    the first place. Output frames are sent to the next
    coroutine for later use in the process of creating diff frames.
    """
    filter_size = len(filt)
    q_blur = Queue(maxsize=filter_size)
    non_zero_count = 0
    frame_count = 0
    avg_count = 0
    idx = 0
    weight = filt[idx]

    try:
        while True:
            frm = (yield)

            if frame_count == 0:
                total = np.zeros_like(zeroframe)

            frame_count += 1
            wt_fr = {weight: frm}
            if not q_blur.full():
                q_blur.put(wt_fr)
                total += frm * weight
                if np.sum(frm) != 0:
                    non_zero_count += weight
                idx += 1
                if idx < filter_size:
                    weight = filt[idx]

            if q_blur.full():
                if non_zero_count != 0:
                    avg = total / non_zero_count
                else:
                    avg = total
                diff_coroutine.send(avg)
                avg_count += 1

                old = q_blur.get()
                old_wt = list(old.keys())[0]
                old_fr = old.get(old_wt)
                total -= old_fr * old_wt
                if np.sum(old_fr) != 0:
                    non_zero_count -= old_wt
                prev_idx = 0
                for i, wt_dict in enumerate(list(q_blur.queue)):
                    prev_wt = list(wt_dict.keys())[0]
                    this_fr = wt_dict.get(prev_wt)
                    total -= this_fr * prev_wt
                    frame_not_zero = (np.sum(this_fr) != 0)
                    if frame_not_zero:
                        non_zero_count -= prev_wt

                    new_wt = filt[i]
                    wt_dict.clear()
                    wt_dict[new_wt] = this_fr

                    total += this_fr * new_wt
                    if frame_not_zero:
                        non_zero_count += new_wt

                    prev_idx = i
                weight = filt[prev_idx + 1]

    except GeneratorExit:
        print("Done with filter after upsample!")


def create_diff_frames(orig_queue, writer_coroutine):
    """
    Coroutine that consumes avgProxy frames, and adds them to one
    queue. The other queue (for original frames) gets populated early,
    so the two queues will fill up at different rates, but eventually
    should have the same amount of frames pass through each of
    them. This method gets the earliest frame from each, and
    uses those to create the diff image, which then gets sent
    off to the coroutine that writes the frames to a file.
    """
    avg_queue = Queue(maxsize=-1)
    created_diff_count = 0
    frame_count = 0
    cumulative = 0
    try:
        while True:
            frm = (yield)
            frame_count += 1
            avg_queue.put(frm)
            if not orig_queue.empty():
                orig = orig_queue.get()
                avg = avg_queue.get()
                cumulative += 1
                diff = create_diff_img(orig, avg)
                writer_coroutine.send(diff)
                created_diff_count += 1
    except GeneratorExit:
        while not orig_queue.empty() and not avg_queue.empty():
            orig = orig_queue.get()
            avg = avg_queue.get()
            diff = create_diff_img(orig, avg)
            writer_coroutine.send(diff)
            created_diff_count += 1

        print("Done with create_diff_frames!")


def create_blur_diff_weighted(input_obj, frame_stride, level_to_create, year=None):
    """
    Creates blur file and diff file(s) for one level of the
    temporal pyramid for a dataset, using a blur filter with weights.
    Precondition: input data covers one day (24 hr time period).
    """

    lvl_to_read = level_to_create - 1

    zero_fr = input_obj.get_black_frame_img()

    blur_path = input_obj.get_blur_path(level_to_create, year)
    writer_blur = set_up_writer(blur_path)
    print("blur_path:", blur_path)

    diff_path = input_obj.get_diff_path(level_to_create, year)
    writer_diff = set_up_writer(diff_path)
    print("diff_path:", diff_path)

    blur_filter = input_obj.get_blur_filter(level_to_create)

    padding_size = frame_stride // 2
    q_orig = Queue(maxsize=-1)

    wf_diff = write_to_file(writer_diff, diff_path)
    next(wf_diff)

    cdf = create_diff_frames(q_orig, wf_diff)
    next(cdf)

    fs1 = filter_stride_1(blur_filter, cdf, zero_fr)
    next(fs1)

    upf = upsample_and_pad(frame_stride, blur_filter, fs1)
    next(upf)

    wf_blur = write_to_file(writer_blur, blur_path)
    next(wf_blur)

    cfaf = create_filtered_avg_frames(blur_filter, frame_stride, wf_blur, upf, zero_fr)
    next(cfaf)

    read_original_frames(input_obj, lvl_to_read, padding_size, q_orig, cfaf, year)


def laplacian_temporal_pyr(in_data):
    """
    Creates all possible levels of the laplacian temporal pyramid for
    input data.
    Precondition: input data covers one day (24 hr time period).
    """
    result_path = in_data.get_result_path()
    if not os.path.exists(result_path):
        print("creating directory for", result_path)
        os.mkdir(result_path)

    print("running laplacian temporal pyramid program")
    strides = in_data.get_stride_sizes()
    level = 1
    for i in strides:
        if i > 0:
            readfile = in_data.get_blur_path(level-1)
            blurfile = in_data.get_blur_path(level)
            baselvl, _f, _s = in_data.get_base_level()
            if (os.path.exists(readfile) or level == baselvl+1) and not os.path.exists(blurfile):
                print("creating pyramid level:", level, "frame stride:", i)
                create_blur_diff_weighted(in_data, i, level)
        elif i == -1:
            print("creating pyramid level:", level, "by copying unaltered frames from prior level.")
            readfile_blur = in_data.get_blur_path(level-1)
            writefile_blur = in_data.get_blur_path(level)
            shutil.copyfile(readfile_blur, writefile_blur)
            readfile_diff = in_data.get_diff_path(level-1)
            writefile_diff = in_data.get_diff_path(level)
            shutil.copyfile(readfile_diff, writefile_diff)
        level += 1


def stitch_vids_together(input_vid_list, writer):
    """
    This method concatenates video files and saves to outpath.
    precondition: input_vid_list is sorted
    """
    for filename in input_vid_list:
        print("stitching:", filename)
        frames = skvideo.io.vreader(filename)
        for frm in frames:
            writer.writeFrame(frm)

    frames.close()
    writer.close()


def stitch_level(in_data, level, year=None):
    """
    Stitches together all of the blur and diff files from a particular level,
    into one file (or multiple files if total length of time > 4 hours).
    """
    blurpath = in_data.get_blur_path(level, year)
    if not os.path.exists(blurpath):
        blurs = in_data.get_multi_blur_globstring(level, year)
        blur_file_list = glob.glob(blurs)
        blur_file_list.sort()
        writer_blur = set_up_writer(blurpath)
        stitch_vids_together(blur_file_list, writer_blur)
        writer_blur.close()

    else:
        print("stitched blur file already exists for level", level)

    diffpath = in_data.get_diff_path(level, year)
    if not os.path.exists(diffpath):
        diffs = in_data.get_multi_diff_globstring(level, year)
        diff_file_list = glob.glob(diffs)
        diff_file_list.sort()
        writer_diff = set_up_writer(diffpath)
        stitch_vids_together(diff_file_list, writer_diff)
        writer_diff.close()

    else:
        print("stitched diff file already exists for level", level)


def create_pyr_multiday(in_data, year):
    """
    First, this function stitches together all available level 15 blur files
    and all available level 15 diff files. Level 15 = 1 framer per 24 hrs.
    Then, it continues building levels of the pyramid using the stitched
    together multi-day blur file as the base. It stops building levels once
    it has create a blur file with only one frame.
    NOTE: this function requires a year to be specified (as a 4-digit integer),
    and will only create the pyramid up to level 21, for that one year.
    """
    level = 15
    stitch_level(in_data, level, year)

    finished = False
    while not finished:
        lvl_to_create = level + 1
        stride = in_data.get_stride(lvl_to_create)
        create_blur_diff_weighted(in_data, stride, lvl_to_create, year)
        curr_blur_file = in_data.get_blur_path(lvl_to_create, year)
        if os.path.exists(curr_blur_file):
            reader = skvideo.io.FFmpegReader(curr_blur_file)
            num_frames, _, _, _ = reader.getShape()
            if num_frames == 1:
                finished = True
        else:
            print("Missing blur file:", curr_blur_file)
        reader.close()
        level +=1


def create_pyr_multiyear(in_data):
    """
    First, this function stitches together all available level 21 blur files
    and all available level 21 diff files. Level 21 = 1 framer per year.
    Then, it continues building levels of the pyramid using the stitched
    together multi-year blur file as the base. It stops building levels once
    it has create a blur file with only one frame.
    """
    for level in range(15, 22):
        stitch_level(in_data, level)

    level = 21
    finished = False
    while not finished:
        lvl_to_create = level + 1
        stride = in_data.get_stride(lvl_to_create)
        create_blur_diff_weighted(in_data, stride, lvl_to_create)
        curr_blur_file = in_data.get_blur_path(lvl_to_create)
        if os.path.exists(curr_blur_file):
            reader = skvideo.io.FFmpegReader(curr_blur_file)
            num_frames, _, _, _ = reader.getShape()
            if num_frames == 1:
                finished = True
        else:
            print("Missing blur file:", curr_blur_file)
        reader.close()
        level +=1


def upsample_pad_for_recon(frame_gen, stride, filt):
    """
    Generator for pyramid reconstruction.
    Upsamples and pads frames from blur file and yields new frames
    to then get filtered.
    """
    filter_size = len(filt)
    padding_size = filter_size // 2
    front_pad_count = padding_size
    last_frame = []

    for frm in frame_gen:
        if frm.dtype == 'uint8':
            frm = frm.astype('float32') / 255

        if len(last_frame) == 0:
            last_frame.append(frm)
        else:
            last_frame[0] = frm

        while front_pad_count > 0:
            front_pad = frm / 2
            yield front_pad
            front_pad_count -= 1

        for i in range(stride):
            yield frm

    last_pad = last_frame[0] / 2
    if filter_size % 2 == 0:
        padding_size -= 1
    for j in range(padding_size):
        yield last_pad


def filtered_frames_for_recon(blur_filter, upsampled, zeroframe):
    """
    Generator very similar to the coroutine filter_stride_1, which
    filters the frames for reconstructing the pyramid, the same
    as how the frames were filtered for construction of the pyramid.
    Uses a generator (upsampled) to get frames, and then yields
    filtered frames back out.
    """
    filter_size = len(blur_filter)
    q_blur = Queue(maxsize=filter_size)

    total = np.zeros_like(zeroframe)
    non_zero_count = 0

    frame_count = 0
    avg_count = 0
    idx = 0
    weight = blur_filter[idx]

    for frm in upsampled:
        frame_count += 1
        wt_fr = {weight: frm}
        if not q_blur.full():
            q_blur.put(wt_fr)
            total += frm * weight
            if np.sum(frm) != 0:
                non_zero_count += weight
            idx += 1
            if idx < filter_size:
                weight = blur_filter[idx]

        if q_blur.full():
            if non_zero_count != 0:
                avg = total / non_zero_count
            else:
                avg = total

            yield avg
            avg_count += 1

            old = q_blur.get()
            old_wt = list(old.keys())[0]
            old_fr = old.get(old_wt)
            total -= old_fr * old_wt
            if np.sum(old_fr) != 0:
                non_zero_count -= old_wt
            prev_idx = 0
            for i, wt_dict in enumerate(list(q_blur.queue)):
                prev_wt = list(wt_dict.keys())[0]
                this_fr = wt_dict.get(prev_wt)
                total -= this_fr * prev_wt
                frame_not_zero = (np.sum(this_fr) != 0)
                if frame_not_zero:
                    non_zero_count -= prev_wt

                new_wt = blur_filter[i]
                wt_dict.clear()
                wt_dict[new_wt] = this_fr

                total += this_fr * new_wt
                if frame_not_zero:
                    non_zero_count += new_wt

                prev_idx = i
            weight = blur_filter[prev_idx + 1]


def reconstruction_generator(blur_generator, diff_generator):
    """
    Creates a frame generator for a reconstructed level
    All in-frames and out-frames are normalized float values
    """
    blur_count = 0
    diff_count = 0
    for blur_frame in blur_generator:
        blur_count += 1
        diff_frame = next(diff_generator)
        diff_count += 1
        sum_frame = blur_frame + diff_frame
        sum_frame = np.clip(sum_frame, 0, 1)
        yield sum_frame


def reconstruct_pyr(in_data, top, bottom, subtract_levels):
    """
    Recombines levels of the laplacian temporal pyramid, with option to have each level
    weighted differently. Weights is a tuple that defaults to all one's, which should
    reconstruct the original set of videos or images (or close to it).
    """
    top_level = top
    bottom_level = bottom
    stride = in_data.get_stride(top_level)
    outpath = in_data.get_reconstruct_path(bottom_level, top_level, subtract_levels)
    writer = set_up_writer(outpath)
    write_count = 0

    blur_filter = in_data.get_blur_filter(top_level)
    top_blur = in_data.get_blur_path(top_level)
    zero_fr = in_data.get_black_frame_img()
    blur_reader = skvideo.io.vreader(top_blur)

    upsampled = upsample_pad_for_recon(blur_reader, stride, blur_filter)
    filtered = filtered_frames_for_recon(blur_filter, upsampled, zero_fr)
    if top_level in subtract_levels:
        weight = 0
    else:
        weight = 1
    diffs_gen = in_data.get_diff_frames(top_level, weight)
    recon_blur = reconstruction_generator(filtered, diffs_gen)
    for level in range(top_level - 1, bottom_level, -1):
        if level > bottom_level:
            print("reconstructing level:", level)
            if level in subtract_levels:
                weight = 0
            else:
                weight = 1

            stitch_level(in_data, level)

            stride = in_data.get_stride(level)
            blur_filter = in_data.get_blur_filter(level)
            upsampled = upsample_pad_for_recon(recon_blur, stride, blur_filter)
            filtered = filtered_frames_for_recon(blur_filter, upsampled, zero_fr)
            diffs_gen = in_data.get_diff_frames(level, weight)
            recon_blur = reconstruction_generator(filtered, diffs_gen)
            if level == bottom_level + 1:
                part = 0
                for frm in recon_blur:
                    frm = (frm * 255).astype('uint8')
                    if (write_count % 432000 == 0) and level == 1:
                        part += 1
                        if part > 1:
                            writer.close()
                        if part <= 6:
                            outpath = in_data.get_reconstruct_path(
                                    bottom_level,
                                    top_level,
                                    subtract_levels,
                                    part
                            )
                            writer = set_up_writer(outpath)
                    write_count += 1
                    if write_count % 100 == 0:
                        print("writing frame of reconstruction:", write_count)
                    writer.writeFrame(frm)

    if top_level == bottom_level + 1:
        stitch_level(in_data, top_level)
        part = 0
        for frm in recon_blur:
            frm = (frm * 255).astype('uint8')
            if write_count % 432000 == 0 and bottom_level == 1:
                part += 1
                if part > 1:
                    writer.close()
                if part <= 6:
                    outpath = in_data.get_reconstruct_path(
                            bottom_level,
                            top_level,
                            subtract_levels,
                            part
                    )
                    writer = set_up_writer(outpath)
            write_count += 1
            if write_count % 100 == 0:
                print("writing frame of reconstruction:", write_count)
            writer.writeFrame(frm)
        writer.close()
    else:
        writer.close()
