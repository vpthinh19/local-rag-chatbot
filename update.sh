#!/bin/bash

source .venv/bin/activate

mkdir -p models
if command -v hf &> /dev/null; then
    models_count=$(find models -maxdepth 1 -type f -name "*.gguf" -printf "1\n" | wc -l)

    if [[ -d models ]] && [[ $models_count == 4 ]]; then
        echo "Models cached (found $models_count files), checking for updates..."
    else
        echo "Models not found or incomplete (found $models_count/3 files), downloading..."
    fi
    hf download unsloth/gemma-4-E4B-it-qat-GGUF \
        gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
        mtp-gemma-4-E4B-it.gguf \
        --local-dir models
    
    hf download gpustack/bge-m3-GGUF \
        bge-m3-Q8_0.gguf \
        --local-dir models
    
    hf download gpustack/bge-reranker-v2-m3-GGUF \
        bge-reranker-v2-m3-Q8_0.gguf \
        --local-dir models
else
    echo "Error: 'hf' command not found. Please install huggingface-hub."
    exit 1
fi