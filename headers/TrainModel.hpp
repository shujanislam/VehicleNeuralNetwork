#ifndef TRAINMODEL_HPP
#define TRAINMODEL_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include "DetectType.hpp"
#include "DataSet.hpp"

using namespace std;

extern map<string, float> car_weight;
extern map<string, float> bike_weight;
extern vector<pair<string, string>> train_data;
extern vector<string> car_words;
extern vector<string> bike_words;
// Forward declarations for Detect functions
float DetectCar(string input);
float DetectBike(string input);

// TrainModel function
inline void TrainModel() {
    float learning_rate = 0.1;

    // Initialize all words in weight maps
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        while (ss >> word) {
            if (car_weight.find(word) == car_weight.end()) car_weight[word] = 0.0;
            if (bike_weight.find(word) == bike_weight.end()) bike_weight[word] = 0.0;
        }
    }

    // Train car weights
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        float car_score = DetectCar(sample.first);
        float target_car = (sample.second.find("car") != string::npos) ? 1.0 : 0.0;
        float error_car = target_car - car_score;

        while (ss >> word) {
            for (auto &w : car_words) {
                if (w == word) {
                    car_weight[word] += learning_rate * error_car;
                }
            }
        }
    }

    // Train bike weights
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        float bike_score = DetectBike(sample.first);
        float target_bike = (sample.second.find("bike") != string::npos) ? 1.0 : 0.0;
        float error_bike = target_bike - bike_score;

        while (ss >> word) {
            for (auto &w : bike_words) {
                if (w == word) {
                    bike_weight[w] += learning_rate * error_bike;
                }
            }
        }
    }
}

#endif // TRAINMODEL_HPP
