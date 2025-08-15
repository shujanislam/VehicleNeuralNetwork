#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<map>
#include<algorithm>
#include "headers/TrainModel.hpp"
#include "headers/DetectType.hpp"

using namespace std;

map<string, float> car_weight;
map<string, float> bike_weight;

// Hidden layer neurons for each category
float hidden1_car = 0.0, hidden2_car = 0.0, hidden3_car = 0.0;
float hidden1_bike = 0.0, hidden2_bike = 0.0, hidden3_bike = 0.0;

int main() {
    for(int epoch = 0; epoch < 20; epoch++){
      TrainModel();
    }

    string input;
    cout << "Enter your input: ";
    getline(cin, input);

    float car_score = DetectCar(input);
    float bike_score = DetectBike(input);

    if (car_score > bike_score) {
        cout << "Prediction: Car" << endl << "Score: "<< car_score << endl;
    } else if (bike_score > car_score) {
        cout << "Prediction: Bike" << endl << "Score: " << bike_score << endl;
    } else {
        cout << "Prediction: None" << endl;
    }

    return 0;
}
