#include <csv.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

int main(){
    csv engine;

    vector<vector<float>> data = engine.readCSV("data/BostonHousing.csv");
    engine.printCSV();
}