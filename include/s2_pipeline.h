#pragma once
// s2_pipeline.h — End-to-end TTS pipeline
//
// Orchestrates: tokenize → encode reference → build prompt → generate → decode → WAV

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <string>

namespace s2 {

struct PipelineParams {
    // Paths
    std::string model_path;       // unified GGUF
    std::string tokenizer_path;   // tokenizer.json

    // Input
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;

    // Generation
    GenerateParams gen;

    // Backend
    int32_t npu_device = -1;   // -1 = CPU only
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Load model + tokenizer + codec
    bool init(const PipelineParams & params);

    // Run synthesis: text (+ optional reference audio) → WAV
    bool synthesize(const PipelineParams & params);

    // Run synthesis: text (+ optional reference audio) → wav_buffer
    bool synthesize_to_memory(const PipelineParams & params, void** wav_buffer, size_t* wav_size);

    // Run synthesis: text (+ optional reference audio) → Vector
    bool synthesize_raw(const PipelineParams & params, std::vector<float> & audio_out);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool initialized_ = false;
};

} // namespace s2
