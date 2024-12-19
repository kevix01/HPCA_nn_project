//
// Created by kevin on 18/12/24.
//

#include "dataset_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

DatasetLoader::DatasetLoader(const std::string& file_path, const std::string& labels_file_path, char delimiter, bool label_first, bool ignore_first_feature, const std::unordered_map<std::string, int>& label_map)
    : file_path(file_path), labels_file_path(labels_file_path), delimiter(delimiter), label_first(label_first), ignore_first_feature(ignore_first_feature), label_map(label_map) {}

void DatasetLoader::load() {
    // Load main data file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    size_t expected_feature_count = 0;
    bool first_line = true;

    while (std::getline(file, line)) {
        std::vector<std::string> tokens = split(line, delimiter);
        if (tokens.empty()) continue;

        if (first_line) {
            expected_feature_count = tokens.size();
            first_line = false;
        } else {
            if (tokens.size() != expected_feature_count) {
                std::cerr << "Inconsistent number of features at line: " << line << std::endl;
                continue;
            }
        }

        std::vector<float> sample;
        size_t start_idx = ignore_first_feature ? 1 : 0;

        if (label_first && start_idx < tokens.size()) {
            const std::string& label_str = tokens[start_idx];
            if (label_map.find(label_str) != label_map.end()) {
                labels.push_back(label_map.at(label_str));
                ++start_idx; // Move past the label
            } else {
                std::cerr << "Invalid label: " << label_str << std::endl;
                continue;
            }
        }

        for (size_t i = start_idx; i < tokens.size(); ++i) {
            try {
                if (!label_first && i == tokens.size() - 1 && label_map.find(tokens[i]) != label_map.end()) {
                    labels.push_back(label_map.at(tokens[i]));
                } else {
                    sample.push_back(std::stof(tokens[i]));
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid input: " << tokens[i] << " at line: " << line << std::endl;
                continue;
            }
        }
        features.push_back(sample);
    }

    file.close();

    if (!labels_file_path.empty()) {
        loadLabelsFromFile(labels_file_path);
    }
}


void DatasetLoader::normalizeFeatures() {
    if (features.empty()) return;

    size_t num_features = features[0].size();
    std::vector<float> min_values(num_features, std::numeric_limits<float>::max());
    std::vector<float> max_values(num_features, std::numeric_limits<float>::lowest());

    // Find min and max values for each feature
    for (const auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (sample[i] < min_values[i]) min_values[i] = sample[i];
            if (sample[i] > max_values[i]) max_values[i] = sample[i];
        }
    }

    // Normalize the features
    for (auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (max_values[i] != min_values[i]) { // Avoid division by zero
                sample[i] = (sample[i] - min_values[i]) / (max_values[i] - min_values[i]);
            } else {
                sample[i] = 0; // If min and max are the same, set feature to 0
            }
        }
    }
}


void DatasetLoader::loadLabelsFromFile(const std::string& labels_file_path) {
    std::ifstream labels_file(labels_file_path);
    if (!labels_file.is_open()) {
        std::cerr << "Failed to open labels file: " << labels_file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(labels_file, line)) {
        if (!line.empty() && label_map.find(line) != label_map.end()) {
            labels.push_back(label_map.at(line));
        }
    }

    labels_file.close();
}

std::vector<std::string> DatasetLoader::split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<std::vector<float>> DatasetLoader::getFeatures() const {
    return features;
}

std::vector<int> DatasetLoader::getLabels() const {
    return labels;
}



