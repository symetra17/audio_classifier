from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import cv2
import os
import glob

words = ['up', 'down', 'left', 'right', 'on', 'off', 'yes', 'no', 'three', 'four', 'five', 'six']

def get_feature(inp, outname):
        (rate,sig) = wav.read(inp)
        feat = logfbank(sig,rate,nfilt=8*26, lowfreq=100, nfft=2048)
        feat = np.where(feat<-6, -6, feat)
        print(feat.shape, feat.max(), feat.min())
        feat = feat + 6
        img = feat/feat.max()
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=0)
        img = np.rot90(img, axes=(0,1))
        img = img * 255
        cv2.imwrite(outname, img.astype(np.uint8))


if __name__=='__main__':

    for word in words:
      try:
        os.mkdir(os.path.join('images', word))
      except:
        pass

      files = glob.glob(os.path.join(R'C:\Users\dva\Music\voice_command', word, '*.wav'))
      for inp in files:
        fname_body = os.path.split(inp)[-1]
        fname_body = os.path.splitext(fname_body)[0]
        outname = os.path.join('images', word, fname_body+'.png')
        get_feature(inp, outname)
