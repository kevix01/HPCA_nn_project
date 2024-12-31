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

// Constructor for DatasetLoader
DatasetLoader::DatasetLoader(const std::string& file_path, const std::string& labels_file_path, char delimiter, bool label_first, bool ignore_first_feature, const std::unordered_map<std::string, int>& label_map)
    : file_path(file_path), labels_file_path(labels_file_path), delimiter(delimiter), label_first(label_first), ignore_first_feature(ignore_first_feature), label_map(label_map) {}

// Load the dataset from the file
void DatasetLoader::load() {
    // Open the main data file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    size_t expected_feature_count = 0; // Expected number of features per sample
    bool first_line = true; // Flag to check if it's the first line of the file

    // Read the file line by line
    while (std::getline(file, line)) {
        // Split the line into tokens based on the delimiter
        std::vector<std::string> tokens = split(line, delimiter);
        if (tokens.empty()) continue; // Skip empty lines

        // Check the number of features in the first line
        if (first_line) {
            expected_feature_count = tokens.size();
            first_line = false;
        } else {
            // Ensure consistency in the number of features for subsequent lines
            if (tokens.size() != expected_feature_count) {
                std::cerr << "Inconsistent number of features at line: " << line << std::endl;
                continue;
            }
        }

        std::vector<float> sample; // Vector to store features of the current sample
        size_t start_idx = ignore_first_feature ? 1 : 0; // Index to start reading features

        // Handle the label if it's the first column
        if (label_first && start_idx < tokens.size()) {
            const std::string& label_str = tokens[start_idx];
            if (label_map.find(label_str) != label_map.end()) {
                labels.push_back(label_map.at(label_str)); // Map label string to integer and store
                ++start_idx; // Move past the label to read features
            } else {
                std::cerr << "Invalid label: " << label_str << std::endl;
                continue;
            }
        }

        // Process the remaining tokens as features
        for (size_t i = start_idx; i < tokens.size(); ++i) {
            try {
                // Handle the label if it's the last column
                if (!label_first && i == tokens.size() - 1 && label_map.find(tokens[i]) != label_map.end()) {
                    labels.push_back(label_map.at(tokens[i])); // Map label string to integer and store
                } else {
                    sample.push_back(std::stof(tokens[i])); // Convert token to float and add to features
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid input: " << tokens[i] << " at line: " << line << std::endl;
                continue;
            }
        }
        features.push_back(sample); // Add the sample to the features vector
    }

    file.close(); // Close the file

    // Load labels from a separate file if provided
    if (!labels_file_path.empty()) {
        loadLabelsFromFile(labels_file_path);
    }
}

// Normalize the features to the range [0, 1]
void DatasetLoader::normalizeFeatures() {
    if (features.empty()) return; // Skip if there are no features

    size_t num_features = features[0].size(); // Number of features per sample
    std::vector<float> min_values(num_features, std::numeric_limits<float>::max()); // Vector to store min values
    std::vector<float> max_values(num_features, std::numeric_limits<float>::lowest()); // Vector to store max values

    // Find the min and max values for each feature
    for (const auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (sample[i] < min_values[i]) min_values[i] = sample[i];
            if (sample[i] > max_values[i]) max_values[i] = sample[i];
        }
    }

    // Normalize each feature to the range [0, 1]
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

// Load labels from a separate file
void DatasetLoader::loadLabelsFromFile(const std::string& labels_file_path) {
    std::ifstream labels_file(labels_file_path); // Open the labels file
    if (!labels_file.is_open()) {
        std::cerr << "Failed to open labels file: " << labels_file_path << std::endl;
        return;
    }

    std::string line;
    // Read the file line by line
    while (std::getline(labels_file, line)) {
        if (!line.empty() && label_map.find(line) != label_map.end()) {
            labels.push_back(label_map.at(line)); // Map label string to integer and store
        }
    }

    labels_file.close(); // Close the file
}

// Split a string into tokens based on a delimiter
std::vector<std::string> DatasetLoader::split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens; // Vector to store tokens
    std::string token;
    std::istringstream tokenStream(line); // Use a string stream to split the line
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token); // Add non-empty tokens to the vector
        }
    }
    return tokens;
}

// Getter for features
std::vector<std::vector<float>> DatasetLoader::getFeatures() const {
    return features;
}

// Getter for labels
std::vector<int> DatasetLoader::getLabels() const {
    return labels;
}



