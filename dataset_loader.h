//
// Created by kevin on 18/12/24.
//

#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <vector>
#include <string>
#include <unordered_map>

class DatasetLoader {
public:
    DatasetLoader(const std::string& file_path, const std::string& labels_file_path = "", char delimiter = ',', bool label_first = false, bool ignore_first_feature = false, const std::unordered_map<std::string, int>& label_map = {});

    void load();

    std::vector<std::vector<float>> getFeatures() const;
    std::vector<int> getLabels() const;
    void normalizeFeatures();

private:
    std::string file_path;
    std::string labels_file_path;
    char delimiter;
    bool label_first;
    bool ignore_first_feature;
    std::unordered_map<std::string, int> label_map;
    std::vector<std::vector<float>> features;
    std::vector<int> labels;

    void loadLabelsFromFile(const std::string& labels_file_path);
    std::vector<std::string> split(const std::string& line, char delimiter);
};

#endif // DATASET_LOADER_H








