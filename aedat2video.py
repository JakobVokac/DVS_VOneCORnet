
from datetime import time
import struct
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.utils import FormatReplace

V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def loadaerdat(datafile, length=0, version=V2, debug=0, camera='DVS128'):
    """Load AER data file and parse these properties of AE events.
    Source:
    https://sourceforge.net/p/jaer/code/HEAD/tree/scripts/python/jAER_utils/loadaerdat.py
    Paramters
    ---------
    datafile : string
        absolute path to the file to read
    length : int
        how many bytes(B) should be read; default 0=whole file
    version : string
        which file format version is used: "aedat" = v2, "dat" = v1 (old)
    debug : int
        0 = silent, 1 (default) = print summary, >=2 = print all debug
    camera : string
        'DVS128' or 'DAVIS240'
    Returns
    -------
    timestamps : int
        time tamps (in us)
    xaddr : int
        x posititon
    yaddr : int
        y position
    pol : int
        polarity
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if (version == V1):
        print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    if debug > 0:
        print ("file size", length)

    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        #  or str(lt)[:4] == "# cr"
        if str(lt)[:4] == "#End":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            break
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        if debug >= 2:
            print (str(lt))
        # continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    if debug > 0:
        print (xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS

        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (
                   len(timestamps), len(timestamps) / float(10 ** 6),
                   (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (
                   timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    timestamps = np.array(timestamps)
    xaddr = np.array(xaddr)
    yaddr = np.array(yaddr)
    pol = np.array(pol)
    return timestamps, xaddr, yaddr, pol

def gen_dvs_frames(timestamps, xaddr, yaddr, pol, num_frames, time_step=0, fs=3,
                   platform="linux2", device="DAVIS240"):
    """Generate DVS frames from recording.
    Paramters
    ---------
    timestamps : numpy.ndarray
        time stamps record
    xaddr : numpy.ndarray
        x position of event recordings
    yaddr : numpy.ndarry
        y position of event recordings
    pol : nujmpy.ndarray
        polarity of event recordings
    num_frames : int
        number of frames in original video sequence
    fs : int
        maximum of events of a pixel
    platform : string
        recording platform of the source. Available option:
        "macosx", "linux2"
    device : string
        DVS camera model - "DAVIS240" (default), "DVS128", "ATIS"
    Returns
    -------
    frames : list
        list of DVS frames
    fs : int
        a scale factor for displaying the frame
    ts : list
        a list that records start timestamp for each frame
    """
    base = 0
    max_events_idx = timestamps.shape[0]-1
    if time_step == 0:
        time_step = (timestamps[-1]-timestamps[0])/num_frames
    if device == "DAVIS240":
        base_frame = np.zeros((180, 240), dtype=np.int8)
    elif device == "DVS128":
        base_frame = np.zeros((128, 128), dtype=np.int8)
    elif device == "ATIS":
        base_frame = np.zeros((240, 304), dtype=np.int8)
    else:
        base_frame = np.zeros((180, 240), dtype=np.int8)

    # print ("Average frame time: %i" % (time_step))

    frames = []
    ts = []
    while base < max_events_idx and len(frames) < num_frames:
        ts.append(timestamps[base])
        k = base
        diff = 0
        frame = base_frame.copy()
        while diff < time_step and k < max_events_idx:
            if platform == "linux2":
                if device == "DAVIS240":
                    x_pos = min(239, xaddr[k]-1)
                elif device == "DVS128":
                    x_pos = min(127, xaddr[k]-1)
                elif device == "ATIS":
                    x_pos = min(304, xaddr[k]-1)
            elif platform == "macosx":
                if device == "DAVIS240":
                    x_pos = min(239, 240-xaddr[k])
                elif device == "DVS128":
                    x_pos = min(127, 128-xaddr[k])
            if device == "DAVIS240":
                y_pos = min(179, 180-yaddr[k])
            elif device == "DVS128":
                y_pos = min(127, yaddr[k])
            elif device == "ATIS":
                y_pos = min(240, yaddr[k])

            if pol[k] == 1:
                frame[y_pos, x_pos] = min(fs, frame[y_pos, x_pos]+1)
            elif pol[k] == 0:
                frame[y_pos, x_pos] = max(-fs, frame[y_pos, x_pos]-1)
            k += 1
            diff = int(timestamps[k]-timestamps[base])

        base = k-1
        frames.append(frame)

    return frames, fs, ts

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import time
mypaths = []
mypaths.append("d:/datasets/cifar10dvs/automobile")
mypaths.append("d:/datasets/cifar10dvs/bird")
mypaths.append("d:/datasets/cifar10dvs/cat")
mypaths.append("d:/datasets/cifar10dvs/deer")
mypaths.append("d:/datasets/cifar10dvs/dog")
mypaths.append("d:/datasets/cifar10dvs/frog")
mypaths.append("d:/datasets/cifar10dvs/horse")
mypaths.append("d:/datasets/cifar10dvs/ship")
mypaths.append("d:/datasets/cifar10dvs/truck")


for mypath in mypaths:
    files = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and join(mypath, f).endswith(".aedat"))]

    for f in tqdm(files):
        # t0 = time.process_time_ns()

        timestamps, xaddr, yaddr, pol = loadaerdat(f)
        # print(timestamps)
        idxs = np.argsort(timestamps)
        timestamps = timestamps[idxs]
        # print(timestamps)
        xaddr = xaddr[idxs]
        yaddr = yaddr[idxs]
        pol = pol[idxs]
        idx_cut = len(timestamps)
        # print("Len full: ", idx_cut)
        # print( np.flip(np.arange(0,len(timestamps))))
        # t1 = time.process_time_ns()

        # t2 = time.process_time_ns()

        # print("Len < 1e6: ", idx_cut)
        
        indices_cut = np.where(timestamps < 1e6, True, False)
        # t3 = time.process_time_ns()

        # t4 = time.process_time_ns()
        timestamps = timestamps[indices_cut]
        xaddr = xaddr[indices_cut]
        yaddr = yaddr[indices_cut]
        pol = pol[indices_cut]
        # print("Len_np < 1e6: ", len(timestamps2))

        # plt.boxplot(timestamps,notch=True)
        # plt.show()
        # t5 = time.process_time_ns()

        frames, fs, ts = gen_dvs_frames(timestamps,xaddr,yaddr,pol,num_frames=10,time_step=1e5,device="DVS128")
        # t6 = time.process_time_ns()
        # print("Load time: ", t1-t0)
        # print("Loop time: ", t2-t1)
        # print("np.where time: ", t3-t2)
        # print("Cut time: ", t4-t3)
        # print("Cut_np time: ", t5-t4)
        # print("Frame time: ", t6-t5)

        # exit()
        for i,frame in enumerate(frames):
            img = Image.fromarray(frame, 'L')
            img.save(f[:-6]+"_frame_"+str(i)+".png",format="png")
            