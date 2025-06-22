#include "asciioscilliscope/Projection2D.h"
#include <gtest/gtest.h>

using namespace asciioscilliscope;

TEST(Projection2DTest, TrapezoidBasic) {
    TrapezoidalPyramid geom{2.0f, 2.0f, 4.0f, 4.0f, 1.0f, 1.0f};
    auto mask = Projection2D<float>::projectTrapezoid(4, 5, geom);
    EXPECT_EQ(mask.dimension(0), 4);
    EXPECT_EQ(mask.dimension(1), 5);
    for(int i=0;i<4;i++)
        for(int j=0;j<5;j++) {
            EXPECT_GE(mask(i,j), 0.0f);
            EXPECT_LE(mask(i,j), 1.0f);
        }
}

TEST(Projection2DTest, SingleCell) {
    TrapezoidalPyramid geom{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};
    auto mask = Projection2D<float>::projectTrapezoid(1,1,geom);
    EXPECT_EQ(mask.dimension(0), 1);
    EXPECT_EQ(mask.dimension(1), 1);
}

TEST(Projection2DTest, NarrowToWide) {
    TrapezoidalPyramid geom{1.0f,1.0f,5.0f,5.0f,1.0f,1.0f};
    auto mask = Projection2D<float>::projectTrapezoid(3,7,geom);
    EXPECT_EQ(mask.dimension(0), 3);
    EXPECT_EQ(mask.dimension(1), 7);
}
