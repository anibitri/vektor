// vektor-server: the REST API and the RAG app.

#include <httplib.h>
#include <pthread.h>

#include <csignal>
#include <exception>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <thread>

#include "core/flags.hpp"
#include "server/server.hpp"

namespace {

constexpr const char* kUsage =
    R"(usage: vektor-server [--host 127.0.0.1] [--port 8080] [--data-dir data]
                     [--ollama http://localhost:11434] [--embed-model all-minilm]
                     [--llm-model MODEL] [--ui DIR]
  Serves the REST API, and with --ui the built web UI (ui/dist) at /. Index files are kept in --data-dir (index.vkt, rag.vkt).
  Without --llm-model, /rag/ask is turned off (low-space mode).
)";

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::span<char*> args(argv, static_cast<std::size_t>(argc));
        for (const std::string_view arg : args.subspan(1)) {
            if (arg == "--help" || arg == "-h") {
                std::cout << kUsage;
                return 0;
            }
        }
        vektor::Flags flags(args.subspan(1));
        vektor::ServerConfig config;
        const std::string host = flags.get("host", "127.0.0.1");
        const std::uint64_t port = flags.get_count("port", 8080);
        config.data_dir = flags.get("data-dir", "data");
        config.ollama_url = flags.get("ollama", config.ollama_url);
        config.embed_model = flags.get("embed-model", config.embed_model);
        config.llm_model = flags.get("llm-model", "");
        config.ui_dir = flags.get("ui", "");
        flags.check_all_used();
        if (port == 0 || port > 65535) {
            throw std::invalid_argument("--port must be from 1 to 65535");
        }

        // Ctrl-C and `docker stop` send SIGINT / SIGTERM. Block them in every
        // thread and wait for them in one thread, which then stops the server.
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGINT);
        sigaddset(&signals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &signals, nullptr);

        vektor::Api api(config);
        httplib::Server server;
        api.register_routes(server);
        server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
            std::cout << std::format("{} {} {}\n", req.method, req.path, res.status) << std::flush;
        });
        if (!server.bind_to_port(host, static_cast<int>(port))) {
            std::cerr << std::format("error: cannot listen on {}:{}\n", host, port);
            return 1;
        }
        std::thread([&server, signals] {
            int sig = 0;
            sigwait(&signals, &sig);
            server.stop();
        }).detach();

        std::cout << std::format(
                         "vektor-server on http://{}:{}\n  index: {} vectors, RAG: {} chunks\n"
                         "  Ollama: {} (embeddings: {}, answers: {})\n",
                         host, port, api.index_size(), api.rag_size(), config.ollama_url,
                         config.embed_model,
                         config.llm_model.empty() ? "off (low-space mode)" : config.llm_model)
                  << std::flush;
        server.listen_after_bind();
        std::cout << "stopped\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
