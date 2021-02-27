#include "diabetesBoard.h"

class Watershed
{

public:
    Watershed(){};
    Watershed(Mat markers, int numberOfRegions)
    {
        mMarkers = markers;
        getAllFoodRegionsArea(markers, numberOfRegions);
    };

    Mat runWatershed(Mat *img0, Mat *maskMarkers, Mat *markers, Mat *wshed,Mat imgGray);
    void getAllFoodRegionsArea(Mat markers, int numberOfRegions);
    int getFoodRegionArea(Point point);
    int getFoodRegionIndex(Point point);

private:
    Mat mMarkers;
    int numComp;
    vector<int> foodAreas;
};
