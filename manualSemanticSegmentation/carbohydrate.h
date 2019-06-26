#ifndef CARBO_DETECTOR
#define CARBO_DETECTOR

#include "diabetesBoard.h"

struct foodRegion{
    int regionPixeis;
    float relation;
    float density;
    float weigh;
    float carbo;

    foodRegion(int regionP, float carboRelation, float foodDensity){ //construtor
        regionPixeis = regionP;
        relation = carboRelation;
        density = foodDensity;
    }
};

class carboDetector{

public:

    /*
     * Variaveis
     */
    float totalWeigh;


    /*
     * Métodos
     */

    carboDetector();
    void saveRegionPixeis(int regionPixeis, string name);
    float calculateCarbo();

private:

    float constant;
    float totalCarbo;
    std::map<string,pair<float,float>> foods;
    vector<foodRegion> foodFeatures;
    vector<string> foodNames;


    /*
     * Métodos
     */

    void saveRegionFeaturesFile();




};

#endif
