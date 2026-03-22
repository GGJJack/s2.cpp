#pragma once
// s2_audio.h — WAV/MP3 audio I/O for s2.cpp
//
// Read audio from WAV or MP3 files, resample to target sample rate,
// and write output WAV files.

#include <cstdint>
#include <string>
#include <vector>

#include "../third_party/dr_wav.h"

namespace s2 {

struct AudioData {
    std::vector<float> samples;   // mono interleaved float32
    int32_t            sample_rate = 0;
};

// Read an audio file (WAV or MP3). Returns mono float32.
bool audio_read(const std::string & path, AudioData & out);
bool audio_read_from_memory(const void * in_data, size_t in_data_size, AudioData & out);

// Write mono float32 audio to WAV file.
bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate);
// Write mono float32 audio to WAV.
bool audio_write_memory_wav(void ** pWavData, size_t * pWavSize, const float * data, size_t n_samples, int32_t sample_rate);
void audio_free_memory_wav(void** pWavData, size_t* pWavSize, const drwav_allocation_callbacks* pAllocationCallbacks);

// Resample mono float32 audio from src_rate to dst_rate (simple linear interpolation).
// For production, a polyphase resampler is preferred.
std::vector<float> audio_resample(const float * data, size_t n_samples, int32_t src_rate, int32_t dst_rate);

// Helper wrappers used by the pipeline
bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate = 0);
bool load_audio_from_memory(const void * data, size_t bytes, AudioData & out, int32_t target_sample_rate = 0);
bool save_audio(const std::string & path, const std::vector<float> & data, int32_t sample_rate);

} // namespace s2
