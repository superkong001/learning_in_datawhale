参考： 
- https://github.com/Ucas-HaoranWei/GOT-OCR2.0?tab=readme-ov-file#install
- https://huggingface.co/ucaslcl/GOT-OCR2_0

环境：

- python 3.10
- torch==2.0.1
- torchvision==0.15.2
- transformers==4.37.2
- tiktoken==0.6.0
- verovio==4.3.1
- accelerate==0.28.0
- numpy

```bash
# Clone this repository and navigate to the GOT folder
git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git
cd 'the GOT folder'

# Install Package
conda create -n got python=3.10 -y
conda activate got
pip install numpy
...
pip install -e .

# Install Flash-Attention
pip install ninja
pip install flash-attn --no-build-isolation

# GOT Weights
```
