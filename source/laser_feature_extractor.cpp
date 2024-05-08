
#include "laser_feature_extractor.hpp"

int main( int argc, char **argv )
{
    Param param;
    Laser_feature laser_feature(param);
    laser_feature.run();
    return 0;
}