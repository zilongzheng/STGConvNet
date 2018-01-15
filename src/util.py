from __future__ import division


import os
import numpy as np
import math
from PIL import Image
import scipy.misc
import subprocess

def loadVideoToFrames(data_path, syn_path, ffmpeg_loglevel = 'quiet'):
    videos = [f for f in os.listdir(data_path) if f.endswith(".avi") or f.endswith(".mp4")]
    num_videos = len(videos)

    for i in range(num_videos):
        video_path = os.path.join(data_path, videos[i])
        out_dir = os.path.join(syn_path, "sequence_%d" % i)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            subprocess.call('ffmpeg -loglevel {} -i {} {}/%03d.png'.format(ffmpeg_loglevel,video_path, out_dir), shell=True)
    return num_videos


def cell2img(filename, out_dir='./final_result',image_size=224, margin=2):
    img = scipy.misc.imread(filename, mode='RGB')
    num_cols = img.shape[1] // image_size
    num_rows = img.shape[0] // image_size
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = img[ir*(image_size+margin):image_size + ir*(image_size+margin),
                   ic*(image_size+margin):image_size + ic*(image_size+margin),:]
            scipy.misc.imsave("%s/%03d.png" % (out_dir,ir*num_cols+ic), temp)
    print(img.shape)

def img2cell(images, col_num=10, margin=2):
    [num_images, size_h, size_w, num_channel] = images.shape
    row_num = int(math.ceil(num_images/col_num))
    saved_img = np.zeros(((row_num * size_h + margin * (row_num - 1)),
                          (col_num * size_w + margin * (col_num - 1)),
                          num_channel), dtype=np.float32)
    for idx in range(num_images):
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = np.squeeze(images[idx])
        temp = np.maximum(0.0, np.minimum(255.0, np.round(temp)))
        gLow = temp.min()
        gHigh = temp.max()
        temp = (temp - gLow) / (gHigh - gLow)
        saved_img[(size_h + margin) * ir:size_h + (size_h + margin) * ir,
        (size_w + margin) * ic:size_w + (size_w + margin) * ic, :] = temp
    return saved_img

def getTrainingData(data_path, num_frames=70, image_size=100, isColor=True, postfix='.png'):
    num_channel = 3
    if not isColor:
        num_channel = 1
    videos = [f for f in os.listdir(data_path) if f.startswith('sequence')]
    num_videos = len(videos)
    images = np.zeros(shape=(num_videos, num_frames, image_size, image_size, num_channel))
    for iv in range(num_videos):
        video_path = os.path.join(data_path, 'sequence_%d' % iv)
        imgList = [f for f in os.listdir(video_path) if f.endswith(postfix)]
        imgList.sort()
        imgList = imgList[:num_frames]
        for iI in range(len(imgList)):
            image = Image.open(os.path.join(video_path, imgList[iI])).resize((image_size, image_size), Image.BILINEAR)
            if isColor:
                image = np.asarray(image.convert('RGB')).astype(float)
            else:
                image = np.asarray(image.convert('L')).astype(float)
                image = image[..., np.newaxis]
            images[iv, iI, :,:,:] = image
    return images.astype(float)

def saveSampleVideo(samples, out_dir, original=None, global_step=None, ffmpeg_loglevel='quiet', fps=25):
    [num_video, num_frames, image_size, _, _] = samples.shape

    for iv in range(num_video):
        result_dir = os.path.join(out_dir, 'sequence_%d' % iv)
        if global_step >= 0:
            result_dir = os.path.join(result_dir, "step_%04d" % global_step)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for ifr in range(num_frames):
            saved_img = np.squeeze(samples[iv,ifr, :,:,:])
            saved_img = np.maximum(0.0, np.minimum(255.0, np.round(saved_img)))
            max_val = saved_img.max()
            min_val = saved_img.min()
            saved_img = (saved_img - min_val) / (max_val - min_val)
            scipy.misc.imsave("%s/%03d.png" % (result_dir, ifr), saved_img)
        subprocess.call('ffmpeg -loglevel {} -r {} -i {}/%03d.png -vcodec mpeg4 -y {}/sample.avi'.format(
            ffmpeg_loglevel,fps, result_dir,result_dir),shell=True)

    if len(original) > 0:
        result_dir = os.path.join(out_dir, 'all')
        if global_step >= 0:
            result_dir = os.path.join(result_dir, "step_%04d" % global_step)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for ifr in range(num_frames):
            combined_img = np.concatenate((original, samples), axis=0)
            saved_img = img2cell(np.squeeze(combined_img[:, ifr, :, :, :]), col_num=num_video+1, margin=10)
            scipy.misc.imsave("%s/%03d.png" % (result_dir, ifr), saved_img)
        subprocess.call('ffmpeg -loglevel {} -r {} -i {}/%03d.png -vcodec mpeg4 -y {}/sample.avi'.format(
            ffmpeg_loglevel, fps, result_dir, result_dir), shell=True)


def saveSampleSequence(samples, sample_dir, iter, col_num=10):
    num_video  = samples.shape[0]

    for iv in range(num_video):
        save_dir = os.path.join(sample_dir, "sequence_%d" % iv)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        scipy.misc.imsave("%s/%04d.png" % (save_dir, iter), img2cell(samples[iv], col_num=col_num))