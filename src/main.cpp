#include "../third_party/filesystem.hpp"
namespace fs = ghc::filesystem;
#include "s2_pipeline.h"
#include "s2_server.h"
#include <iostream>
#include <string>
#include <vector>

void print_uso() {
    std::cout << "Usage: s2 [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <path>                  Path to unified GGUF model\n";
    std::cout << "  -t, --tokenizer <path>              Path to tokenizer.json\n";
    std::cout << "  -text <string>                      Text to synthesize\n";
    std::cout << "  -pa, --prompt-audio <p>             Path to reference audio for cloning\n";
    std::cout << "  -pt, --prompt-text <s>              Text of the reference audio for cloning\n";
    std::cout << "  -o, --output <path>                 Output WAV path\n";
    std::cout << "  -v, -c, --vulkan, --cuda <device>   Vulkan/Cuda device index (-1 for CPU)\n";
    std::cout << "  -threads N                          Number of CPU threads for inference (default: 4)\n";
    std::cout << "  -max-tokens N                       Max tokens to generate (default: 512, ~24s audio)\n";
    std::cout << "  -temp F                             Sampling temperature (default: 0.7)\n";
    std::cout << "  -top-p F                            Top-p sampling (default: 0.7)\n";
    std::cout << "  -top-k N                            Top-k sampling (default: 30)\n";
    std::cout << "  --server N                          Start http server\n";
    std::cout << "  -H --host                           Server host\n";
    std::cout << "  -P --port                           Server port\n";
}

inline void init_utf8_console() {
    #ifdef _WIN32
        #include <windows.h>
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif
}

int safe_parse_int(const std::string& str, const std::string& name, int default_value = 0) {
    try {
        return std::stoi(str);
    } catch (...) {
        std::cerr << "Invalid integer for " << name << ": " << str << "\n";
        return default_value;
    }
}

float safe_parse_float(const std::string& str, const std::string& name, float default_value = 0.0f) {
    try {
        return std::stof(str);
    } catch (...) {
        std::cerr << "Invalid float for " << name << ": " << str << "\n";
        return default_value;
    }
}

int main(int argc, char ** argv) {
    init_utf8_console();

    if (argc < 2) {
        print_uso();
        return 1;
    }

    fs::u8arguments u8guard(argc, argv);

    s2::PipelineParams params;
    // Default paths
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.output_path = "out.wav";
    params.text = "Hello world";
    params.gpu_device = -1;
    params.backend_type = -1;

    int32_t use_server = 0;
    s2::ServerParams serverParams;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) params.model_path = argv[++i];
        } else if (arg == "-t" || arg == "--tokenizer") {
            if (i + 1 < argc) params.tokenizer_path = argv[++i];
        } else if (arg == "-text") {
            if (i + 1 < argc) params.text = argv[++i];
        } else if (arg == "-pa" || arg == "--prompt-audio") {
            if (i + 1 < argc) params.prompt_audio_path = argv[++i];
        } else if (arg == "-pt" || arg == "--prompt-text") {
            if (i + 1 < argc) params.prompt_text = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) params.output_path = argv[++i];
        } else if (arg == "-v" || arg == "--vulkan") {
            if (i + 1 < argc) params.gpu_device = safe_parse_int(argv[++i], arg, params.gpu_device); params.backend_type = 0; //Vulkan.
        } else if (arg == "-c" || arg == "--cuda") {
            if (i + 1 < argc) params.gpu_device = safe_parse_int(argv[++i], arg, params.gpu_device); params.backend_type = 1; //Cuda.
        } else if (arg == "-threads") {
            if (i + 1 < argc) params.gen.n_threads = safe_parse_int(argv[++i], arg, params.gen.n_threads);
        } else if (arg == "-max-tokens") {
            if (i + 1 < argc) params.gen.max_new_tokens = safe_parse_int(argv[++i], arg, params.gen.max_new_tokens);
        } else if (arg == "-temp") {
            if (i + 1 < argc) params.gen.temperature = safe_parse_float(argv[++i], arg, params.gen.temperature);
        } else if (arg == "-top-p") {
            if (i + 1 < argc) params.gen.top_p = safe_parse_float(argv[++i], arg, params.gen.top_p);
        } else if (arg == "-top-k") {
            if (i + 1 < argc) params.gen.top_k = safe_parse_int(argv[++i], arg, params.gen.top_k);
        } else if (arg == "--server") {
            if (i + 1 < argc) use_server = safe_parse_int(argv[++i], arg, use_server);
        } else if (arg == "-H" || arg == "--host") {
            if (i + 1 < argc) serverParams.host = argv[++i];
        } else if (arg == "-P" || arg == "--port") {
            if (i + 1 < argc) serverParams.port = safe_parse_int(argv[++i], arg, serverParams.port);
        } else if (arg == "-h" || arg == "--help") {
            print_uso();
            return 0;
        }
    }

    // If tokenizer path was not explicitly set, search for tokenizer.json in:
    //   1. Same directory as the model file
    //   2. Parent directory of the model file
    //   3. Working directory (default fallback)
    if (params.tokenizer_path == "tokenizer.json") {
        std::string model_path = params.model_path;
        size_t slash = model_path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string model_dir = model_path.substr(0, slash + 1);
            // Check same dir as model
            std::string candidate = model_dir + "tokenizer.json";
            if (FILE * f = std::fopen(candidate.c_str(), "r")) {
                std::fclose(f);
                params.tokenizer_path = candidate;
            } else {
                // Check parent dir
                size_t parent_slash = model_dir.find_last_of("/\\", slash - 1);
                if (parent_slash != std::string::npos) {
                    candidate = model_dir.substr(0, parent_slash + 1) + "tokenizer.json";
                    if (FILE * f2 = std::fopen(candidate.c_str(), "r")) {
                        std::fclose(f2);
                        params.tokenizer_path = candidate;
                    }
                }
            }
        }
    }

    if (use_server == 1)
    {
        serverParams.pipeline = params;

        s2::Server server;
        if (!server.serve(serverParams))
        {
            std::cerr << "Server initialization failed." << std::endl;
            return 1;
        }
        
        return 0;
    }

    s2::Pipeline pipeline;
    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }

    if (!pipeline.synthesize(params)) {
        std::cerr << "Synthesis failed." << std::endl;
        return 1;
    }

    return 0;
}
