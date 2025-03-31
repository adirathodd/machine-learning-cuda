#ifndef CSV_H
#define CSV_H
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class csv {
    private:
        std::string filename;
        std::vector<std::vector<float>> data;

        std::vector<float> split(const std::string &line, char delimiter);

    public:
        csv() {};
        ~csv() {};

        std::vector<std::vector<float>> get_data() {
            return data;
        }

        std::vector<std::vector<float>> readCSV(std::string filename);

        void printCSV() {
            for (const auto &row : data) {
                for (const auto &col : row) {
                    std::cout << col << " ";
                }
                std::cout << std::endl;
            }
        }
};

#endif