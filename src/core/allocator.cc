#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        // uint64_t: 8字节无符号整数
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t head_address_offset = 0;
        size_t next_elem_head_offset = 0;
        std::cout << "****cur freeBlocks size = " << freeBlocks.size() << std::endl;
        if (freeBlocks.size() <= 1) {
            if (freeBlocks.empty()) {
                std::cout << "*****************cur freeBlocks is empty" << std::endl;
                freeBlocks[head_address_offset] = size;
                std::cout << "****************in empty, offset = " << head_address_offset << " it's block size = " << size << std::endl;
                peak = head_address_offset + size;
                return head_address_offset;
            }

            if (freeBlocks.size() == 1) {
                auto it = freeBlocks.begin();
                auto block_size = it->second;
                std::cout << "*******************cur size = " << size << " old offset = " << head_address_offset << std::endl;
                head_address_offset = block_size;
                std::cout << "**********new offset = " << head_address_offset << std::endl;
                freeBlocks[head_address_offset] = size;
                peak = head_address_offset + size;
                return head_address_offset;
            }
        } else  {
            auto left = freeBlocks.begin();
            auto right = ++freeBlocks.begin();
            for (; left != freeBlocks.end() && right != freeBlocks.end(); ++left, ++right) {
                auto left_offset = left->first;
                auto left_block_size = left->second;
                auto right_offset = right->first;
                auto right_block_size = right->second;
                // check gap...
                head_address_offset = (left_offset + left_block_size);
                std::cout << "*********left offset = " << left_offset << " left_block_size = " << left_block_size << " right offset = " << right_offset << " right block size = " << right_block_size << " pre alloc size = " << size << std::endl;
                std::cout << "*********head_address_offset + size = " << head_address_offset + size << " right_offset = " << right_offset << std::endl;
                if ((head_address_offset + size) <= right_offset) {
                    freeBlocks[head_address_offset] = size;
                    return head_address_offset;
                }
                // ...check gap
                next_elem_head_offset = (right_offset + right_block_size);
               std::cout << "in for next_elem_head_offset = " << next_elem_head_offset << std::endl;   
            }
            std::cout << "next_elem_head_offset = " << next_elem_head_offset << std::endl;
            freeBlocks[next_elem_head_offset] = size;
            peak = next_elem_head_offset + size;
        }
        return next_elem_head_offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            auto offset = it->first;
            std::cout << "****************free: cur offset = " << offset << " input addr = " << addr << std::endl;
            if (offset == addr) {
                std::cout << "pre erase key = " << offset << std::endl;
                freeBlocks.erase(offset);
                peak -= it->second;
                std::cout << "free done, new map size = " << freeBlocks.size() << std::endl;
                break;
            }
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
