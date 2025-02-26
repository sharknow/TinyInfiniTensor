#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini
{
    TEST(Allocator, testAlloc)
    {
        std::cout << "*******************************start 1 test" << std::endl;
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        std::cout << "*******************************start 1 test 0" << std::endl;
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        size_t offsetA = allocator.alloc(a->getBytes());
        std::cout << "*******************************start 1 test 1" << std::endl;
        size_t offsetB = allocator.alloc(b->getBytes());
        std::cout << "*******************************start 1 test 2" << std::endl;
        size_t offsetC = allocator.alloc(c->getBytes());
        std::cout << "*******************************start 1 test 3" << std::endl;
        // free b, then allocate d
        allocator.free(offsetB, b->getBytes());
        std::cout << "*******************************start 1 test 4" << std::endl;
        size_t offsetD = allocator.alloc(d->getBytes());
        std::cout << "*******************************start 1 test 5" << std::endl;
        // expected to be a->d->c
        EXPECT_EQ(offsetB, offsetD);
        std::cout << "*******************************start 1 test 6" << std::endl;
        ASSERT_FALSE(offsetA == 0 && offsetB == 0 && offsetC == 0 && offsetD == 0);
        std::cout << "*******************************end 1 test" << std::endl;
    }

    TEST(Allocator, testAllocWithEndFreeBlock)
    {
        std::cout << "start test 2" << std::endl;
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d =
            make_ref<TensorObj>(Shape{2, 2, 2, 3}, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        allocator.alloc(a->getBytes());
        std::cout << "start test 2 0" << std::endl;
        allocator.alloc(b->getBytes());
        std::cout << "start test 2 1" << std::endl;
        size_t offsetC = allocator.alloc(c->getBytes());
        std::cout << "start test 2 2" << std::endl;
        allocator.info();
        std::cout << "start test 2 3" << std::endl;
        // free c, then allocate d
        allocator.free(offsetC, c->getBytes());
        std::cout << "start test 2 4" << std::endl;
        size_t offsetD = allocator.alloc(d->getBytes());
        std::cout << "start test 2 5" << std::endl;
        allocator.info();
        std::cout << "start test 2 6" << std::endl;
        // expected to be a->b->d, with no free block between b and c
        EXPECT_EQ(offsetC, offsetD);
        std::cout << "start test 2 7" << std::endl;
    }

    TEST(Allocator, testGetPtr)
    {
        std::cout << "start test 3" << std::endl;
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        std::cout << "start test 3 0" << std::endl;
        Allocator allocator = Allocator(runtime);
        std::cout << "start test 3 1" << std::endl;
        // allocate a->b->c->d
        allocator.alloc(a->getBytes());
        std::cout << "start test 3 2" << std::endl;
        allocator.alloc(b->getBytes());
        std::cout << "start test 3 3" << std::endl;
        allocator.alloc(c->getBytes());
        std::cout << "start test 3 4" << std::endl;
        allocator.alloc(d->getBytes());
        std::cout << "start test 3 5" << std::endl;
        // multiple calls to the getPtr() function should return the same pointer
        void *ptr1 = allocator.getPtr();
        std::cout << "start test 3 6" << std::endl;
        void *ptr2 = allocator.getPtr();
        std::cout << "start test 3 7" << std::endl;
        EXPECT_EQ(ptr1, ptr2);
        std::cout << "done test 3" << std::endl;
    }

} // namespace infini
