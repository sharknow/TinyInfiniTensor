#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    // Same
    if (A.size() == B.size()) {
        size_t i = 0;
        for (; i < A.size(); ++i) {
            if (A[i] != B[i]) {
                break;
            }
        }
        if (i == (A.size() - 1)) {
            return A;
        }
    }

    // Different
    size_t rank_A = A.size(), rank_B = B.size();
    Shape new_shape;
    Shape result_shape;
    if (rank_A > rank_B) {
        for (size_t i = 0; i < (rank_A - rank_B); ++i) {
            new_shape.emplace_back(1);
        }
        for (size_t i = 0; i < rank_B; ++i) {
            new_shape.emplace_back(B[i]);
        }
        // Now: rank_A == rank_B
        for (size_t i = 0; i < new_shape.size(); ++i) {
            result_shape.emplace_back(std::max(A[i] ,new_shape[i]));
        }
    } else if (rank_B > rank_A) {
        for (size_t i = 0; i < (rank_B - rank_A); ++i) {
            new_shape.emplace_back(1);
        }
        for (size_t i = 0; i < rank_A; ++i) {
            new_shape.emplace_back(A[i]);
        }
        // Now: rank_A == rank_B
        for (size_t i = 0; i < new_shape.size(); ++i) {
            result_shape.emplace_back(std::max(B[i] ,new_shape[i]));
        }
    } else {
        // Now: rank_A == rank_B
        for (size_t i = 0; i < rank_A; ++i) {
            result_shape.emplace_back(std::max(A[i] ,B[i]));
        }
    }

    return result_shape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
