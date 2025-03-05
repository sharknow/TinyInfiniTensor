#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini
{
    TEST(Graph, Optimize)
    {
        std::cout << "-----------start test Graph" << std::endl;
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        std::cout << "i1 Guid = " << i1->getGuid() << std::endl;
        Tensor i2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        std::cout << "i2 Guid = " << i2->getGuid() << std::endl;
        Tensor t1 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
        std::cout << "t1 Guid = " << t1->getGuid() << std::endl;
        Tensor t2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        std::cout << "t2 Guid = " << t2->getGuid() << std::endl;
        Tensor t3 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
        std::cout << "t3 Guid = " << t3->getGuid() << std::endl;
        Tensor o = g->addTensor({2, 3, 4, 4}, DataType::UInt32);
        std::cout << "o Guid = " << o->getGuid() << std::endl;
        g->addOpWithOutputs<TransposeObj>(i1, t1, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<TransposeObj>(t1, t2, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<TransposeObj>(i2, t3, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<MatmulObj>(t2, t3, o);
        // 优化前
        std::cout << "----not opti: " << std::endl;
        g->print();
        std::cout << "----opti: " << std::endl;
        g->optimize();
        // 优化后
        std::cout << "----opti 0" << std::endl;
        g->print();
        std::cout << "----opti 1" << std::endl;
        EXPECT_EQ(g->getOperators().size(), 1);
        std::cout << "----opti 2" << std::endl;
        EXPECT_EQ(g->getTensors().size(), 3);
        std::cout << "----opti 3" << std::endl;
        EXPECT_EQ(g->getOperators()[0]->getOpType().underlying(), 7);
        std::cout << "----opti 4" << std::endl;
        auto op = as<MatmulObj>(g->getOperators()[0]);
        EXPECT_EQ(op->getInputs(0)->getGuid(), 2);
        EXPECT_EQ(op->getInputs(1)->getGuid(), 3);
        EXPECT_EQ(op->getOutputs()[0], o);
        EXPECT_EQ(op->getTransA(), false);
        EXPECT_EQ(op->getTransB(), true);
    }
}