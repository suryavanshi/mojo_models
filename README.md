# mojo_models
Mojo Models

Serve Custom Models with Mojo - https://docs.modular.com/max/tutorials/serve-custom-model-architectures

## Usage

After logging in to Hugging Face and setting the token:

```bash
huggingface-cli login --token <TOKEN>
export HUGGINGFACE_HUB_TOKEN=<TOKEN>
pip install timm torchvision
```

Serve the Gemma3n model with MAX:

```bash
max serve --model-path google/gemma-3n-E2B-it --custom-architectures gemma3n
```
