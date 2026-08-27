#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct Model {
    std::vector<std::string> top_tokens;
    std::unordered_map<std::string, std::vector<std::string>> transitions;
};

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open model file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string unescape_json(std::string value) {
    std::string out;
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '\\' && index + 1 < value.size()) {
            const char next = value[++index];
            if (next == 'n') {
                out.push_back('\n');
            } else if (next == 't') {
                out.push_back('\t');
            } else {
                out.push_back(next);
            }
        } else {
            out.push_back(value[index]);
        }
    }
    return out;
}

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

Model parse_model(const std::string& json) {
    Model model;
    const std::regex token_pattern(R"JSON("token"\s*:\s*"((?:\\.|[^"])*)")JSON");
    for (std::sregex_iterator it(json.begin(), json.end(), token_pattern), end; it != end; ++it) {
        model.top_tokens.push_back(unescape_json((*it)[1].str()));
    }

    const std::regex transition_pattern(R"JSON("((?:\\.|[^"])*)"\s*:\s*\[\s*\{\s*"token"\s*:\s*"((?:\\.|[^"])*)")JSON");
    for (std::sregex_iterator it(json.begin(), json.end(), transition_pattern), end; it != end; ++it) {
        model.transitions[unescape_json((*it)[1].str())].push_back(unescape_json((*it)[2].str()));
    }

    if (model.top_tokens.empty()) {
        model.top_tokens = {"function", "(", ")", "{", "const", "result", "=", ";", "return", "}", "console", ".", "log"};
    }
    return model;
}

std::string sanitize_identifier(const std::string& prompt) {
    std::string identifier;
    for (unsigned char c : prompt) {
        if (std::isalnum(c)) {
            identifier.push_back(static_cast<char>(std::tolower(c)));
        } else if (!identifier.empty() && identifier.back() != '_') {
            identifier.push_back('_');
        }
    }
    if (identifier.empty() || std::isdigit(static_cast<unsigned char>(identifier.front()))) {
        identifier = "cam_request_" + identifier;
    }
    if (identifier.size() > 36) {
        identifier.resize(36);
    }
    while (!identifier.empty() && identifier.back() == '_') {
        identifier.pop_back();
    }
    return identifier.empty() ? "cam_request" : identifier;
}

std::vector<std::string> generate_tokens(const Model& model, const std::string& prompt, int limit) {
    std::vector<std::string> output;
    uint64_t state = fnv1a(prompt);
    for (int i = 0; i < limit && !model.top_tokens.empty(); ++i) {
        state = (state * 6364136223846793005ULL) + 1ULL;
        output.push_back(model.top_tokens[state % model.top_tokens.size()]);
    }
    return output;
}

std::string js_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '\\' || c == '\'') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c != '\r') {
            out.push_back(c);
        }
    }
    return out;
}

std::string render_token_stream(const std::vector<std::string>& tokens) {
    std::ostringstream code;
    for (const std::string& token : tokens) {
        if (token == ";" || token == "{" || token == "}") {
            code << token << '\n';
        } else if (token == ")" || token == "]" || token == "." || token == ",") {
            code << token;
            if (token == ",") {
                code << ' ';
            }
        } else if (token == "(" || token == "[") {
            code << token;
        } else {
            code << token << ' ';
        }
    }
    return code.str();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model-json> <prompt> [output-js]\n";
        return 64;
    }

    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    const std::string output_path = argc >= 4 ? argv[3] : "generated/haruharu-rim-cam-output.js";

    try {
        const Model model = parse_model(read_file(model_path));
        const std::string function_name = sanitize_identifier(prompt);
        const std::vector<std::string> sampled = generate_tokens(model, prompt, 96);

        const std::string learned_hint = render_token_stream(sampled);
        std::ostringstream code;
        code << "// Generated locally by HaruharuRIM CAM. No external LLM was used.\n";
        code << "// Prompt: " << prompt << "\n";
        code << "// Learned token hint: " << learned_hint.substr(0, 180) << "\n\n";
        code << "async function " << function_name << "(input = {}) {\n";
        code << "  const camBrand = 'HaruharuRIM CAM';\n";
        code << "  const prompt = '" << js_escape(prompt) << "';\n";
        code << "  const params = new URLSearchParams(input).toString();\n";
        code << "  const endpoint = input.endpoint || '/api/haruharu-rim-cam';\n";
        code << "  const response = await fetch(`${endpoint}${params ? `?${params}` : ''}`);\n";
        code << "  const data = await response.json();\n";
        code << "  const viewModel = { brand: camBrand, prompt, generatedAt: new Date().toISOString(), data };\n";
        code << "  if (typeof document !== 'undefined' && input.target) {\n";
        code << "    const target = document.querySelector(input.target);\n";
        code << "    if (target) target.textContent = JSON.stringify(viewModel, null, 2);\n";
        code << "  }\n";
        code << "  console.log(camBrand, prompt, viewModel);\n";
        code << "  return viewModel;\n";
        code << "}\n\n";
        code << "export { " << function_name << " };\n";

        const std::size_t slash = output_path.find_last_of('/');
        if (slash != std::string::npos) {
            const std::string mkdir_command = "mkdir -p '" + output_path.substr(0, slash) + "'";
            if (std::system(mkdir_command.c_str()) != 0) {
                throw std::runtime_error("Could not create output directory");
            }
        }
        std::ofstream output(output_path);
        output << code.str();
        std::cout << "Generated " << output_path << " from " << model_path << "\n";
    } catch (const std::exception& error) {
        std::cerr << "HaruharuRIM CAM generation failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
