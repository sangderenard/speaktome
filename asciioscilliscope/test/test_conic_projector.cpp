#include "asciioscilliscope/ConicProjector3D.h"
#include <gtest/gtest.h>

using namespace asciioscilliscope;

TEST(ConicProjectorTest, ShapeVerification) {
    int radial = 2;
    int angular = 3;
    int depth = 4;
    auto tensor = ConicProjector3D<float>::projectCone(radial, angular, depth, 0.f);
    EXPECT_EQ(tensor.dimension(0), depth);
    EXPECT_EQ(tensor.dimension(1), radial);
    EXPECT_EQ(tensor.dimension(2), angular);
}
