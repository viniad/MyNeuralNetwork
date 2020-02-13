#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;
int main() {
    NeuralNetwork net(2,2,1, 0.1);
    net.train({0,0}, {0}, 1000);

    vector<double> res = net.predict({0,0});
    for(int i =0; i< res.size(); i++){
        cout << res[i] << endl;
    }
    net.save("Model");
    return 0;
}