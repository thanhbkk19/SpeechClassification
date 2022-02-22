import os
import librosa
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import cv2
def load_data(data_path):
    items = os.listdir(data_path)
    for i in range(5):
        items_path = os.path.join(data_path,items[i])
        print(items_path)
    return
def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    """ Creates a spectrogram of a wav file.
    :param audio_path: path of wav file
    :param save_path:  path of spectrogram to save
    :param spectrogram_dimensions: number of pixels the spectrogram should be. Defaults (64,64)
    :param noverlap: See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    :param cmap: the color scheme to use for the spectrogram. Defaults to 'gray_r'
    :return:
    """
    samples,sr = librosa.load(audio_path)
    pad = lambda a, i : a[0:i] if a.shape[0]>=i else np.hstack((a,np.zeros(i-a.shape[0])))
    samples = pad(samples,10000)
    samples = librosa.amplitude_to_db(np.abs(librosa.stft(samples)),ref=np.max)
    print(samples.shape)
    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    return 0

def save_spectrogram(audio_path, save_path="/home/gumiho/project/SpeechClassification/Spectrogram",spectrogram_dimensions=(64, 64)):
    items = os.listdir(audio_path)
    
    for idx in range(10):
        items_path = os.path.join(audio_path,items[idx])
        save_path_new = os.path.join(save_path,items[idx].replace(".wav",".png"))
        wav_to_spectrogram(items_path,spectrogram_dimensions=spectrogram_dimensions,save_path=save_path_new)
    return 0
class MNIST_data(Dataset):
    def __init__(self, data_path, transform = None,type="train"):
        self.data_path = data_path
        self.transform = transform
        self.type = type
    def __len__(self):
        if self.type == "train":
            return len(os.listdir(self.data_path))-200
        elif self.type =="val":
            return 200
    def __getitem__(self,idx):
        items = os.listdir(self.data_path)
        if self.type =="train":
            items_path = os.path.join(self.data_path,items[idx])
            struct = items[idx].split("_")
        elif self.type =="val":
            items_path = os.path.join(self.data_path,items[idx+len(os.listdir(self.data_path))-200])
            struct = items[idx+len(os.listdir(self.data_path))-200].split("_")

        #struct = items[idx].split("_")
        digit = struct[0]

        # spectrogram = cv2.imread(items_path)
        # spectrogram = np.array(spectrogram,dtype=np.float32)
        wav,sr = librosa.load(items_path)
        #pad = lambda a, i : a[0:i] if a.shape[0]>=i else np.hstack((a,np.zeros(i-a.shape[0])))
        pad1d = lambda a, i : a[0:i] if a.shape[0]>=i else np.hstack((a,np.zeros(i-a.shape[0])))
        pad2d = lambda a,i : a[:,0:i] if a.shape[1]>=i else np.hstack((a,np.zeros((a.shape[0],i-a.shape[1]))))
        samples = pad1d(wav,30000)
        spectrogram = np.abs(librosa.stft(wav))
        padded_spectrogram = pad2d(spectrogram,40)
        spectrogram = np.array(padded_spectrogram,dtype=np.float32)
        spectrogram = np.expand_dims(spectrogram,axis=-1)
        #print(spectrogram.shape)
        label = [0]*10
        label[int(digit)] = 1
        label = np.array(label,dtype=np.float32)
        if self.transform is not None:
            augmentation = self.transform(image=spectrogram,mask=label)
            spectrogram = augmentation["image"]
            label = augmentation["mask"]
        #print(spectrogram.shape)
        return spectrogram, label

def test1():
    """
    function use to convert all dato to Spectrogram image --> Spectrogram folder
    """
    save_spectrogram(audio_path="/home/gumiho/project/SpeechClassification/free-spoken-digit-dataset/recordings",spectrogram_dimensions=(64,1024))
    return
def test2():
    """
    Function use to test image
    """
    import cv2
    img = Image.open("/home/gumiho/project/SpeechClassification/Spectrogram/2_george_12.png")
    print(img)
    return
if __name__ =="__main__":
    items_path = "/home/gumiho/project/SpeechClassification/free-spoken-digit-dataset/recordings/0_george_28.wav"
    samples,sr = librosa.load(items_path)
    pad1d = lambda a, i : a[0:i] if a.shape[0]>=i else np.hstack((a,np.zeros(i-a.shape[0])))
    pad2d = lambda a, i : a[:,0:i] if a.shape[1]>=i else np.hstack((a,np.zeros((a.shape[0],i-a.shape[1]))))
    samples = pad1d(samples,30000)

    spectrogram = np.abs(librosa.stft(samples))

    print(spectrogram.shape)