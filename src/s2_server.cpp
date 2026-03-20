#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
// #define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/s2_server.h"
#include <iostream>

// httplib::SSLServer svr;

using json = nlohmann::json;

bool create_directories_recursive(const std::string& path)
{
    try
    {
        std::filesystem::path dir_path(path);
        dir_path = dir_path.parent_path();

        if (!dir_path.empty())
        {
            std::filesystem::create_directories(dir_path);
        }
        return true;
    }
    catch (const std::filesystem::filesystem_error& ex)
    {
        std::cout << "Error creating directories: " << ex.what() << std::endl;
        return false;
    }
}

namespace s2
{
    Server::Server() {}
    Server::~Server() {}

    bool Server::serve(const ServerParams& params)
    {
        httplib::Server svr;

        s2::Pipeline pipeline;
        if (!pipeline.init(params.pipeline))
        {
            std::cerr << "Pipeline initialization failed." << std::endl;
            return 0;
        }

        svr.set_pre_routing_handler([](const auto& req, auto& res) -> httplib::Server::HandlerResponse {
            auto start = std::chrono::high_resolution_clock::now();

            std::cout << "[START] " << req.method << " " << req.path << std::endl;

            res.set_header("X-Request-Start",
                std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                    start.time_since_epoch()).count()));

            return httplib::Server::HandlerResponse::Unhandled;
            });

        svr.set_logger([](const auto& req, const auto& res)
            {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end.time_since_epoch()).count() -
                    std::stoll(res.get_header_value("X-Request-Start", "0"));

                std::cout << "[END] " << req.method << " " << req.path
                    << " -> " << res.status
                    << " (" << duration << "ms)" << std::endl; });

        svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res)
            {
                PipelineParams pipelineParams;
                pipelineParams.gen = params.pipeline.gen;

                if (!req.form.has_field("text"))
                {
                    json err = { {"error", "No text field in multipart form"} };
                    res.set_content(err.dump(), "application/json");
                    res.status = 400;
                    return;
                }

                pipelineParams.text = req.form.get_field("text");

                if (req.form.has_field("reference_text"))
                {
                    pipelineParams.prompt_text = req.form.get_field("reference_text");
                }

                if (req.form.has_field("params"))
                {
                    try {
                        auto j = json::parse(req.form.get_field("params"));

                        if (j.contains("max_new_tokens")) {
                            int32_t val = j["max_new_tokens"].get<int32_t>();
                            pipelineParams.gen.max_new_tokens = val;
                        }

                        if (j.contains("temperature")) {
                            float val = j["temperature"].get<float>();
                            pipelineParams.gen.temperature = val;
                        }

                        if (j.contains("top_p")) {
                            float val = j["top_p"].get<float>();
                            pipelineParams.gen.top_p = val;
                        }

                        if (j.contains("top_k")) {
                            int32_t val = j["top_k"].get<int32_t>();
                            pipelineParams.gen.top_k = val;
                        }

                        if (j.contains("min_tokens_before_end")) {
                            int32_t val = j["min_tokens_before_end"].get<int32_t>();
                            pipelineParams.gen.min_tokens_before_end = val;
                        }

                        if (j.contains("n_threads")) {
                            int32_t val = j["n_threads"].get<int32_t>();
                            pipelineParams.gen.n_threads = val;
                        }

                        if (j.contains("verbose")) {
                            bool val = j["verbose"].get<bool>();
                            pipelineParams.gen.verbose = val;
                        }

                        if (j.contains("repeat_penalty")) {
                            float val = j["repeat_penalty"].get<float>();
                            pipelineParams.gen.repeat_penalty = val;
                        }
                    }
                    catch (const json::parse_error& e) {
                        json err = { {"error", "JSON parse error"} };
                        res.set_content(err.dump(), "application/json");
                        res.status = 400;
                        return;
                    }
                }


                const void* ref_audio_buffer = nullptr;
                size_t ref_audio_size = 0;

                void* wav_buffer = nullptr;
                size_t wav_size = 0;

                if (req.form.has_file("reference")) {
                    const auto& file = req.form.get_file("reference");

                    if (!file.content.empty()) {
                        ref_audio_buffer = file.content.data();
                        ref_audio_size = file.content.size();
                    }
                }

                if (!pipeline.synthesize_to_memory(pipelineParams, const_cast<void**>(&ref_audio_buffer), &ref_audio_size, &wav_buffer, &wav_size))
                {
                    std::cerr << "Synthesis failed." << std::endl;
                    res.status = 500;
                    res.set_content("Failed to create WAV", "text/plain");
                    return;
                }

                if (!wav_buffer || wav_size == 0)
                {
                    res.status = 500;
                    res.set_content("Failed to create WAV", "text/plain");
                    return;
                }

                res.set_content(
                    static_cast<const char*>(wav_buffer),
                    wav_size,
                    "audio/wav"
                );
                res.set_header("Content-Disposition", "attachment; filename=\"generated_audio.wav\"");
                res.status = 200;

                audio_free_memory_wav(&wav_buffer, &wav_size, nullptr); });

        std::cout << "Server starting on http://" << params.host << ":" << params.port << "..." << std::endl;

        svr.listen(params.host.c_str(), params.port);

        return 1;
    }

}
