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
        std::cout << "****cur freeBlocks size = " << freeBlocks.size() << std::endl;
        if (freeBlocks.size() <= 1) {
            if (freeBlocks.empty()) {
                std::cout << "*****************cur freeBlocks is empty" << std::endl;
                freeBlocks[head_address_offset] = size;
                std::cout << "****************in empty, offset = " << head_address_offset << " it's block size = " << size << std::endl;
                return head_address_offset;
            }

            if (freeBlocks.size() == 1) {
                auto it = freeBlocks.begin();
                auto block_size = it->second;
                std::cout << "*******************cur size = " << size << " old offset = " << head_address_offset << std::endl;
                head_address_offset += block_size;
                std::cout << "**********new offset = " << head_address_offset << std::endl;
                freeBlocks[head_address_offset] = size;
                return head_address_offset;
            }
        } else  {
            auto left = freeBlocks.begin();
            auto right = ++freeBlocks.begin();
            for (; left != freeBlocks.end() && right != freeBlocks.end(); ++left, ++right) {
                auto left_offset = left->first;
                auto left_block_size = left->second;
                auto right_offset = right->first;
                head_address_offset += (left_offset + left_block_size);
                std::cout << "*********left offset = " << left_offset << " left_block_size = " << left_block_size << " right offset = " << right_offset << std::endl;
                if ((head_address_offset + size) <= right_offset) {
                    freeBlocks[head_address_offset] = size;
                    return head_address_offset;
                }
            }
            freeBlocks[head_address_offset] = size;
        }
        return head_address_offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        std::cout << "***************freeBlocks size = " << freeBlocks.size() << " want free addr = " << addr << " want free size = " << size << std::endl;
        std::cout << "map info: " << std::endl;
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            std::cout << "offset = " << it->first << " block_size = " << it->second << std::endl;
        }
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            auto offset = it->first;
            std::cout << "****************free: cur offset = " << offset << " input addr = " << addr << std::endl;
            if (offset == addr) {
                std::cout << "pre erase key = " << offset << std::endl;
                freeBlocks.erase(offset);
                std::cout << "free done, new map size = " << freeBlocks.size() << std::endl;
                break;
            }
        }
        std::cout << "free done, now map size = " << freeBlocks.size() << std::endl;
        std::cout << "done free, map info: " << std::endl;
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            std::cout << "offset = " << it->first << " block_size = " << it->second << std::endl;
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
