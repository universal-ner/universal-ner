## Training

This directory contains the training code for **UniversalNER**, which is adapted from the [FastChat](https://github.com/lm-sys/FastChat) library. 

###  **Changes Made:**
- 📝 Added the conversation-style instruction tuning template.
- 🚀 Enhanced the training code to support lazy data processing.
- 🐛 Fixed the bug leading to OOM (Out of Memory) during model saving.

### **License**
The training code is licensed under the **Apache 2.0** License.

### **Training**
1. Install the package:
```bash
pip3 install -e ".[model_worker,webui]"
```

2. Download UniversalNER data `train.json` from [Huggingface](https://huggingface.co/Universal-NER).

3. Run the training script:
```bash
sh train.sh
```

Note: Our model is trained with 8 A100 40G GPUs. If you encounter an OOM error during model saving, you can find solutions [here](https://github.com/pytorch/pytorch/issues/98823).
