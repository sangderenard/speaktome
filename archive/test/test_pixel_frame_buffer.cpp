#include "asciioscilliscope/PixelFrameBuffer.h"
#include <gtest/gtest.h>

using namespace asciioscilliscope;

TEST(PixelFrameBufferTest, DiffComputation) {
    PixelFrameBuffer<float> buf(1,1,1,2,2);
    Eigen::Tensor<float,5> data(1,1,1,2,2);
    data.setZero();
    data(0,0,0,0,0) = 0.5f;
    data(0,0,0,0,1) = 1.0f;
    data(0,0,0,1,0) = -0.5f;
    data(0,0,0,1,1) = -1.0f;
    buf.updateRender(data);

    auto events = buf.getDiffAndSwap(0.0f);
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(std::get<0>(events[0]), 0);
    EXPECT_EQ(std::get<1>(events[0]), 0);
    EXPECT_EQ(std::get<2>(events[0]), 0);
    // Coordinates are row-major traversal
    // check one value
    EXPECT_FLOAT_EQ(std::get<5>(events[0]), data(0,0,0,0,0));

    // No changes on second call
    buf.updateRender(data);
    auto noEvents = buf.getDiffAndSwap(0.0f);
    EXPECT_TRUE(noEvents.empty());
}
