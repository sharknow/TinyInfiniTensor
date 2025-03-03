#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini
{
    using ExpectOutput = vector<float>;

    TEST(Matmul, ShapeInference)
    {   
        std::cout << "++++++++++++start test 1" << std::endl;
        auto runtime = NativeCpuRuntimeObj::getInstance();
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{1, 3, 5});
            auto B = g->addTensor(Shape{1, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{1, 3, 2}));
        }
        std::cout << "++++++++++++start test 2" << std::endl;
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{3, 5, 4});
            auto B = g->addTensor(Shape{3, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{3, 4, 2}));
        }
        std::cout << "++++++++++++start test 3" << std::endl;
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{1, 2, 3, 5});
            auto B = g->addTensor(Shape{1, 1, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{1, 2, 3, 2}));
        }
        std::cout << "++++++++++++start test 4" << std::endl;
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{2, 3, 5, 4});
            auto B = g->addTensor(Shape{1, 3, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{2, 3, 4, 2}));
        }
        std::cout << "++++++++++++start test 5" << std::endl;
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{2, 3, 5, 4});
            auto B = g->addTensor(Shape{1, 3, 2, 5});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{2, 3, 4, 2}));
        }
        std::cout << "++++++++++++end test" << std::endl;
    }

}; // namespace infini