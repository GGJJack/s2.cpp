#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
// #define CPPHTTPLIB_OPENSSL_SUPPORT
#include "../include/s2_server.h"
#include <iostream>

// httplib::SSLServer svr;

using json = nlohmann::json;

using s2::GenerateParams;
struct GenerateParamsRequest : GenerateParams
{
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t min_tokens_before_end;
    int32_t n_threads;
    bool verbose;
    float repeat_penalty;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GenerateParamsRequest,
    max_new_tokens, temperature, top_p, top_k,
    min_tokens_before_end, n_threads, verbose, repeat_penalty)

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
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    end.time_since_epoch()).count() -
                    std::stoll(res.get_header_value("X-Request-Start", "0"));

                std::cout << "[END] " << req.method << " " << req.path
                    << " -> " << res.status
                    << " (" << duration << "μs)" << std::endl; });

        svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res)
            {

                if (!req.form.has_file("reference")) {
                    json err = { {"error", "No file field in multipart form"} };
                    res.set_content(err.dump(), "application/json");
                    res.status = 400;
                    return;
                }

                s2::PipelineParams pipelineParams;

                if (!req.form.has_field("text"))
                {
                    json err = { {"error", "No text field in multipart form"} };
                    res.set_content(err.dump(), "application/json");
                    res.status = 400;
                }

                std::cout << "  " << "text: " << req.form.get_field("text") << std::endl;
                pipelineParams.text = req.form.get_field("text");

                if (req.form.has_field("reference_text"))
                {
                    std::cout << "  " << "reference_text: " << req.form.get_field("reference_text") << std::endl;
                    pipelineParams.prompt_text = req.form.get_field("reference_text");
                }

                if (req.form.has_field("params"))
                {
                    try {
                        auto j = json::parse(req.form.get_field("params"));
                        pipelineParams.gen = j.get<GenerateParamsRequest>();
                    }
                    catch (const json::parse_error& e) {
                        json err = { {"error", "JSON parse error"} };
                        res.set_content(err.dump(), "application/json");
                        res.status = 400;
                        return;
                    }
                }

                const auto& file = req.form.get_file("reference");

                auto safe_filename = file.filename.empty() ? "unnamed_file" : httplib::sanitize_filename(file.filename);
                std::string temp_dir_path = "./TEMP_AUDIO/";
                std::string reference_audio_path = temp_dir_path + safe_filename;

                pipelineParams.prompt_audio_path = reference_audio_path;

                if (!create_directories_recursive(temp_dir_path)) {
                    res.status = 500;
                    res.set_content(R"({"error": "Cannot create temp directory"})", "application/json");
                    return;
                }

                std::ofstream ofs(reference_audio_path, std::ios::binary);
                if (!ofs)
                {
                    res.status = 500;
                    res.set_content(R"({"error": "Cannot write file"})", "application/json");
                    return;
                }

                ofs.write(file.content.data(), file.content.size());
                ofs.close();

                void* wav_buffer = nullptr;
                size_t wav_size = 0;
                if (!pipeline.synthesize_to_memory(pipelineParams, &wav_buffer, &wav_size))
                {
                    std::cerr << "Synthesis failed." << std::endl;
                    res.status = 500;
                    res.set_content("Failed to create WAV", "text/plain");
                    return;
                }

                std::filesystem::remove(reference_audio_path);


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
