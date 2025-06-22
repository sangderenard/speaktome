#include "asciioscilliscope/Projection2D.h"
#include <gtest/gtest.h>

using namespace asciioscilliscope;

TEST(Projection2DTest, TrapezoidBasic) {
    auto mask = Projection2D<float>::projectTrapezoid(4, 5, 2.0f, 4.0f);
    // Row 0: width=2 centered in 5 cols => cols 1-2 true
    EXPECT_FALSE(mask(0,0)); EXPECT_TRUE(mask(0,1)); EXPECT_TRUE(mask(0,2)); EXPECT_FALSE(mask(0,3));
    // Row 3: width=4 centered => cols 0-3 true
    EXPECT_TRUE(mask(3,0)); EXPECT_TRUE(mask(3,3)); EXPECT_FALSE(mask(3,4));
}

TEST(Projection2DTest, SingleCell) {
    auto mask = Projection2D<float>::projectTrapezoid(1,1,1.0f,1.0f);
    EXPECT_TRUE(mask(0,0));
}

TEST(Projection2DTest, NarrowToWide) {
    auto mask = Projection2D<float>::projectTrapezoid(3,7,1.0f,5.0f);
    // Row 0 width=1 => center at col3
    EXPECT_TRUE(mask(0,3)); EXPECT_FALSE(mask(0,2));
    // Row2 width=5 => center at col1-5
    EXPECT_TRUE(mask(2,1)); EXPECT_TRUE(mask(2,5)); EXPECT_FALSE(mask(2,0));
}
