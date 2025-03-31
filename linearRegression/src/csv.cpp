#include <csv.h>
#include <sstream>

std::vector<float> csv::split(const std::string &line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    std::vector<float> row;

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    for( const auto &token : tokens ) {
        try {
            row.push_back(std::stof(token));
        } catch (const std::exception &e) {
            row.push_back(0.0f);
            std::cerr << "Error converting token to float: " << token << std::endl;
        }
    }

    return row;
}

std::vector<std::vector<float>> csv::readCSV(std::string filename) {
    this->filename = filename;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        data.push_back(split(line, ','));
    }

    file.close();
    return data;
}