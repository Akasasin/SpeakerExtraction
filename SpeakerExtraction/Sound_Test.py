# _*_ coding : utf-8_*_
# @Time : 2024/12/27 20:44
# @Author : Akasasin
# @File : Sound_Test
# @Project : SpeakerExtraction
# 读取音频文件
import soundfile as sf
if __name__ == '__main__':
    data, samplerate = sf.read('./datasets/wav/tr/mix/ABAC009S0012W0311.wav_BAC009S0012W0275.wav')

    # 打印音频信息
    print(f"音频数据: {data}")
    print(f"采样率: {samplerate} Hz")


