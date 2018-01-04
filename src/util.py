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
            subprocess.call('ffmpeg -loglevel {} -i {} {}/%03d.jpg'.format(ffmpeg_loglevel,video_path, out_dir), shell=True)
    return num_videos


def cell2img(filename, out_dir='./final_result',image_size=100, margin_syn=2):
    img = scipy.misc.imread(filename, mode='RGB')
    num_cols = img.shape[1] // image_size
    num_rows = img.shape[0] // image_size
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = img[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            scipy.misc.imsave("%s/%03d.jpg" % (out_dir,ir*num_cols+ic), temp)
    print(img.shape)



def getTrainingData(data_path, num_frames=70, image_size=100, isColor=True, postfix='.jpg'):
    num_channel = 3
    if not isColor:
        num_channel = 1
    videos = [f for f in os.listdir(data_path) if f.startswith('sequence')]
    num_videos = len(videos)
    images = np.zeros(shape=(num_videos, num_frames, image_size, image_size, num_channel))
    for iv in range(num_videos):
        video_path = os.path.join(data_path, 'sequence_%d' % iv)
        imgList = [f for f in os.listdir(video_path) if f.endswith(postfix)]
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

def saveSampleVideo(samples, out_dir, ffmpeg_loglevel='quiet'):
    [num_video, num_frames, image_size, _, _] = samples.shape

    for iv in range(num_video):
        result_dir = os.path.join(out_dir, 'sequence_%d' % iv)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for ifr in range(num_frames):
            saved_img = np.squeeze(samples[iv,ifr, :,:,:])
            scipy.misc.imsave("%s/%03d.jpg" % (result_dir, ifr), saved_img)
        subprocess.call('ffmpeg -loglevel {} -i {}/%03d.jpg -vcodec mpeg4 -y {}/sample.avi'.format(
            ffmpeg_loglevel,result_dir,result_dir),shell=True)


def saveSampleSequence(samples, sample_dir, iter, col_num=10, margin_syn=2):
    [num_video, num_frames, image_size, _, num_channel] = samples.shape

    row_num = int(math.ceil(num_frames/col_num))
    for iv in range(num_video):
        save_dir = os.path.join(sample_dir, "sequence_%d" % iv)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saved_img = np.zeros(((row_num * image_size + margin_syn * (row_num - 1)),
                              (col_num * image_size + margin_syn * (col_num - 1)),
                              num_channel), dtype=float)
        for ifr in range(num_frames):
            ir = int(math.floor(ifr/col_num))
            ic = ifr % col_num
            temp = np.squeeze(samples[iv,ifr,:,:,:])
            gLow = np.min(temp, axis=(0, 1, 2))
            gHigh = np.max(temp, axis=(0, 1, 2))
            temp = (temp - gLow) / (gHigh - gLow)
            saved_img[(image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
                    (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp
        scipy.misc.imsave("%s/%04d.png" % (save_dir, iter), saved_img)


if __name__ == '__main__':
    #data_path = '/home/zilong/Documents/STGConvNet/output/fire_pot/observed_sequence'
    data_path = '/home/zilong/Documents/STGConvNet/output/sea/synthesis_sequence/sequence_2/1000.png'
    cell2img(data_path, '/home/zilong/Documents/STGConvNet/output/sea/final_result/sequence_2')
    #print(getTrainingData(data_path).shape)