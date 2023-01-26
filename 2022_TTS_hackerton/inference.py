from src.models.tacotron2.text import text_to_sequence
import src.models.utils as models
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from train import parse_args
from src.utils.common.utils import load_wav_to_torch
from src.utils.common.layers import TacotronSTFT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']  
                  
        model.load_state_dict(state_dict)
   
    model.eval()

    if fp16_run:
        model.half()

    return model

# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths

def prepare_input_sequence(texts, cpu_run=False):
    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['korean_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths

def load_mel(path):
    stft = TacotronSTFT()
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != 16000:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / 32768.0 # hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec
        
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    
    fig.savefig('results.png')

def save_mel_for_hifigan(mel):
    print(mel.shape)
    print(type(mel))
    mel = mel.float().data.cpu().numpy()[0]
    np.save('result.npy', mel)
    
    
    
def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x
    
def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    
    # VAE Tacotron2 Inference
    # '분노':1,
    # '기쁨':4,
    # '슬픔':5,
    # '무감정':0,
    # '놀람':6
    # "선수소개/경기진행" : 2,
    # "일반설명" : 3,
    # "득점상황" : 7    
    
    emotion_ids = {'0': 0, '1': 1, '4': 2, '5': 3, '6': 4}
    speaker_ids = {'001': 0, '002': 1, '003': 2, '004': 3, '005': 4, '006': 5, '007': 6, '009': 7, '011': 8, '012': 9, '013': 10, 
                   '014': 11, '015': 12, '016': 13, '020': 14, '021': 15, '022': 16, '023': 17, '024': 18, '026': 19, '027': 20, 
                   '028': 21, '029': 22, '034': 23, '036': 24, '037': 25}
    
    # 변수 선언
    checkpoint_path = "output/checkpoint_Tacotron2_150.pt"
    input_text_file_path = "infer_text_single.txt"
    is_fp16 = True
    is_cpu = False
    emotion_id = '1'
    speaker_id = '006'
    ref_mel_path = "../dataset/multi_speaker_emotion_dataset_2022/any/splitted/A/006/A-A1-A-006-0001.wav"
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    tacotron2 = load_and_setup_model('Tacotron2', parser, checkpoint_path,
                                     is_fp16, is_cpu, forward_is_infer=True) # forward is infer를 해줌으로써 tacotron model의 infer로 간다.
    
    jitted_tacotron2 = torch.jit.script(tacotron2)

    texts = []
    
    emotion_id = to_gpu(torch.IntTensor([emotion_ids[emotion_id]])).long()
    speaker_id = to_gpu(torch.IntTensor([speaker_ids[speaker_id]])).long()
    
    try:
        f = open(input_text_file_path, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    ref_mel = load_mel(ref_mel_path)
    
    measurements = {}
    sequences_padded, input_lengths = prepare_input_sequence(texts, is_cpu)
    
    with torch.no_grad():
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths, ref_mel)

    plot_data((mel.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))  
    
    save_mel_for_hifigan(mel)
      

if __name__ == '__main__':
    main()