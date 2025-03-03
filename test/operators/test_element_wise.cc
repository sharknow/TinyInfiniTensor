#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

    TEST(ElementWise, ShapeInference)
    {
        std::cout << "start test ShapeInference" << std::endl;
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
            Tensor i1 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
        }
        std::cout << "end test ShapeInference" << std::endl;
    }

    TEST(ElementWise, Broadcasting)
    {
        std::cout << "start test Broadcasting" << std::endl;
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
            Tensor i1 = g->addTensor({}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 5}));
        }

        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
            Tensor i1 = g->addTensor({5}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 5}));
        }

        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({4, 5}, DataType::UInt32);
            Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 5}));
        }

        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({1, 4, 5}, DataType::UInt32);
            Tensor i1 = g->addTensor({2, 3, 1, 1}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 5}));
        }

        {
            Graph g = make_ref<GraphObj>(runtime);
            Tensor i0 = g->addTensor({3, 4, 5}, DataType::UInt32);
            Tensor i1 = g->addTensor({2, 1, 1, 1}, DataType::UInt32);
            auto op = g->addOp<AddObj>(i0, i1, nullptr);
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 5}));
        }
        std::cout << "end test Broadcasting" << std::endl;
    }

} // namespace infini
