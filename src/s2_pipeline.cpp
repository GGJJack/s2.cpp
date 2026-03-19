#include "../include/s2_pipeline.h"
#include <iostream>

namespace s2 {

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    std::cout << "--- Pipeline Init ---" << std::endl;
    if (!tokenizer_.load(params.tokenizer_path)) {
        std::cerr << "Pipeline error: could not load tokenizer from " << params.tokenizer_path << std::endl;
        return false;
    }

    if (!model_.load(params.model_path, params.npu_device)) {
        std::cerr << "Pipeline error: could not load model from " << params.model_path << std::endl;
        return false;
    }

    // Codec runs only twice per synthesis (encode ref audio + decode output),
    // not in the hot generation loop — always keep on CPU to save VRAM.
    if (!codec_.load(params.model_path, /*vulkan_device=*/-1)) {
        std::cerr << "Pipeline error: could not load codec from " << params.model_path << std::endl;
        return false;
    }

    // Sync tokenizer config from model hparams so that semantic token IDs,
    // codebook count, and vocab size are consistent between generation and
    // prompt-building regardless of what the tokenizer.json says.
    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    std::vector<float> audio_out;
    if (!this->synthesize_raw(params, audio_out)) {
        std::cerr << "Pipeline error: decode failed." << std::endl;
        return false;
    }

    if (!save_audio(params.output_path, audio_out, codec_.sample_rate())) {
        std::cerr << "Pipeline error: save_audio failed to " << params.output_path << std::endl;
        return false;
    }

    std::cout << "Saved audio to: " << params.output_path << std::endl;
    return true;
}

bool Pipeline::synthesize_to_memory(const PipelineParams & params, void** wav_buffer, size_t* wav_size) {
    std::vector<float> audio_out;
    if (!this->synthesize_raw(params, audio_out)) {
        std::cerr << "Pipeline error: decode failed." << std::endl;
        return false;
    }

    if (!audio_write_memory_wav(
        wav_buffer,
        wav_size,
        audio_out.data(),
        audio_out.size(),
        codec_.sample_rate()
    )) {
        std::cerr << "Pipeline error: audio_write_memory_wav failed" << std::endl;
        return false;
    }

    std::cout << "Audio synthesized" << std::endl;
    return true;
}

bool Pipeline::synthesize_raw(const PipelineParams & params, std::vector<float>& audio_out) {
    if (!initialized_) {
        std::cerr << "Pipeline not initialized." << std::endl;
        return false;
    }

    std::cout << "--- Pipeline Synthesize ---" << std::endl;
    std::cout << "Text: " << params.text << std::endl;

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    // 1. Audio Prompt Loading
    // encode() returns codes in row-major (num_codebooks, T_prompt) format,
    // matching the layout expected by build_prompt() (prompt_codes[c*T+t]).
    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    if (!params.prompt_audio_path.empty()) {
        std::cout << "Loading reference audio: " << params.prompt_audio_path << std::endl;
        AudioData ref_audio;
        if (load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                               params.gen.n_threads, ref_codes, T_prompt)) {
                std::cerr << "Pipeline warning: encode failed, running without reference audio." << std::endl;
                ref_codes.clear();
                T_prompt = 0;
            }
        } else {
            std::cerr << "Pipeline warning: load_audio failed, running without reference audio." << std::endl;
        }
    }

    // 2. Build Prompt Tensor
    // build_prompt expects prompt_codes as (num_codebooks, T_prompt) row-major,
    // which is exactly the format produced by encode() above.
    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, params.prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    // 3. Setup KV Cache
    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!model_.init_kv_cache(max_seq_len)) {
        std::cerr << "Pipeline error: init_kv_cache failed." << std::endl;
        return false;
    }

    // 4. Generate
    // generate() returns GenerateResult.codes in row-major (num_codebooks, n_frames).
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
    if (res.n_frames == 0) {
        std::cerr << "Pipeline error: generation produced no frames." << std::endl;
        return false;
    }

    // 5. Decode
    // codec_.decode() receives codes in row-major (num_codebooks, n_frames),
    // which matches GenerateResult.codes layout.
    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
        std::cerr << "Pipeline error: decode failed." << std::endl;
        return false;
    }
    
    return true;
}

} // namespace s2
