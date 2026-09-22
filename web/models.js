// Single source of truth for where the model files live. `?local=1` on the
// page URL switches to the same local files used during development;
// without it, everything loads from Hugging Face.
export const LOCAL = new URLSearchParams(location.search).get('local') === '1';

const HF_QWEN_GGUF = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main';
const HF_QWEN_TOKENIZER = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main';
const HF_LORA = 'https://huggingface.co/idle-intelligence/llm-of-life-lora/resolve/main';
const HF_STENCIL = 'https://huggingface.co/idle-intelligence/stencil-life/resolve/main';

export const MODEL_URLS = LOCAL ? {
  gguf: './models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf',
  tokenizer: './models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json',
  loraANorules: './lora-a-norules-300.bin',
  loraARules: './lora-a-rules-300.bin',
  loraB16: './lora-b-16-s3-300.bin',
  loraB32: './lora-b-32.bin',
  bert: './bert-d16-L1.bin',
  vecMlp: './mlp2-32-lrfix.bin',
  vecStencil: './stencil-d16-L1.bin',
} : {
  gguf: `${HF_QWEN_GGUF}/qwen2.5-0.5b-instruct-q4_0.gguf`,
  tokenizer: `${HF_QWEN_TOKENIZER}/tokenizer.json`,
  loraANorules: `${HF_LORA}/lora-a-norules-300.bin`,
  loraARules: `${HF_LORA}/lora-a-rules-300.bin`,
  loraB16: `${HF_LORA}/lora-b-16-s3-300.bin`,
  loraB32: `${HF_LORA}/lora-b-32.bin`,
  bert: `${HF_STENCIL}/bert-d16-L1.bin`,
  vecMlp: `${HF_STENCIL}/mlp2-32-lrfix.bin`,
  vecStencil: `${HF_STENCIL}/stencil-d16-L1.bin`,
};
