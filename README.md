<br/>
<h2 align="center">Telespeech-asr-python</h2>
<br/>


[TeleSpeech-ASR（星辰超多方言语音识别大模型）](https://github.com/Tele-AI/TeleSpeech-ASR)是由中国电信人工智能研究院（TeleAI）发布业内首个支持30种方言自由混说的语音识别大模型。

首先感谢电信团队的开源奉献，该模型是目前来看修改版的data2vec， 整个模型类似于wav2vec_ctc， 期待后续技术报告及论文的发布。

由于原项目依赖fairseq和kaldi预处理， 光跑起来就非常麻烦，本项目提供一个不依赖与fairseq和kaldi的**推理环境**方便模型测试。

模型使用官方在KeSpeech数据集8种方言微调的模型

现sherpa-onnx已支持telespeech的c++ runtime， 见[详情](https://github.com/k2-fsa/sherpa-onnx/pull/970)。

## 如何使用

### 1. 安装依赖

torch版runtime需要安装kaldifest和requirements.txt里面的依赖
kaldifest 安装参看 [官方文档](https://github.com/csukuangfj/kaldifeat)

```bash
pip install -r requirements.txt
```


onnxruntime 只需要安装requirements-onnxruntime.txt里面的依赖即可
```bash
pip install -r requirements-onnxruntime.txt
```


### 2. 下载模型

由于本人修改该模型中的键值key，删掉了checkpoint的多余信息，因此本项目不兼容官方原版checkpoint

从huggingface
```bash
wget https://huggingface.co/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt

# 或者使用镜像
wget https://hf-mirror.com/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt
```

### 3. 模型导出

1. torchscript 导出

```bash
PYTHONPATH=$PWD python telespeechasr/torchscript/torchscript_export.py --model_path /path/torch_checkpoint.pt \
--output_dir /path/output_dir
```
2. onnx 导出

```bash
PYTHONPATH=$PWD python telespeechasr/onnx/onnx_export.py --model_path /path/torch_checkpoint.pt
--output_dir /path/output_dir
```

### 4. 模型推理（目前还不支持batch解码）

**以下模型都可在huggingface [下载](https://huggingface.co/lovemefan/telespeech/tree/main)**

1. torch推理， 支持cpu, cuda, mps
```bash

PYTHONPATH=$PWD python telespeechasr/torch/infer.py --model_path /path/finetune_large_kespeech.pt --audio_path /path/audio.wav
```
2. torchscript 推理， 支持cpu, cuda, mps

```bash
PYTHONPATH=$PWD python telespeechasr/torchscript/torchscript_infer.py --model_path /path/model_export_torchscript.pt
--audio_path /path/audio.wav
--device cpu
```

3. onnx 推理, 支持gpu，cpu推理
```bash
PYTHONPATH=$PWD python telespeechasr/onnx/onnx_infer.py --model_path /path/model_export.onnx
--audio_path /path/audio.wav
```
