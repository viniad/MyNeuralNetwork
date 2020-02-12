#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"

typedef struct{
    vector<double> input;
    vector<double> target;
}TrainingData;

typedef struct{
    TrainingData td;
}Data;

using namespace std;
int main() {
    Data data[4];
    data[0].td.input = {0,0};
    data[1].td.input = {0,1};
    data[2].td.input = {1,0};
    data[3].td.input = {1,1};

    data[0].td.target = {0};
    data[1].td.target = {1};
    data[2].td.target = {1};
    data[3].td.target = {0};



    NeuralNetwork net(2,2,1);


    net.train({0,0}, {0}, 1000);
    vector<double> res = net.predict({0,0});
    for(int i =0; i< res.size(); i++){
        cout << res[i] << endl;
    }
    net.save("Model");
    return 0;
}