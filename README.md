# Gemma4-Medical-SFT
a small demo to fine-tune gemma-4-31B for the medical domain

```bash
docker run -it --rm --gpus '"device=0"' -p 8000:8000 -v /opt/gpudata/models:/models vllm/vllm-openai:latest /models/google/gemma-4-31B-it --enforce-eager --served-model-name "gemma4-base" --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja --default-chat-template-kwargs '{"enable_thinking": true}'
```

```bash
docker run -it --rm --gpus '"device=1"' -p 8001:8000 -v /home/songs1/Gemma4-Medical-SFT/train/gemma4-31b-medical-o1:/model vllm/vllm-openai:latest /model --enforce-eager --served-model-name "gemma4-tune" --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja --default-chat-template-kwargs '{"enable_thinking": true}'
```
