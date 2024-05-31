<br/>
<h2 align="center">Telespeech-asr-python</h2>
<br/>


[TeleSpeech-ASR（星辰超多方言语音识别大模型）) ](https://github.com/Tele-AI/TeleSpeech-ASR)是有中国电信人工智能研究院（TeleAI）发布业内首个支持30种方言自由混说的语音识别大模型。

实际上该模型与语言模型不太一致， 只是修改版的data2vec， 整个模型类似于wav2vec_ctc。

由于原项目依赖fairseq和kaldi预处理， 光跑起来就非常麻烦，本项目提供一个不依赖与fairseq和kaldi的运行时方便模型测试。

模型使用官方在KeSpeech数据集8种方言微调的模型

## 如何使用

### 1. 安装kaldifest和依赖

参看 [官方文档](https://github.com/csukuangfj/kaldifeat)

```bash
pip install -r requirements.txt
```

### 2. 下载模型

从huggingface
```bash
wget https://huggingface.co/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt

# 或者使用镜像
wget https://hf-mirror.com/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt
```

### 推理

```bash

PYTHONPATH=$PWD python telespeechasr/torch/infer.py --model_path /path/finetune_large_kespeech.pt --audio_path /path/audio.wav
```

