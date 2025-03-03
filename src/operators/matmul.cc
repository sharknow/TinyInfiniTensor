#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        std::cout << "++++++++++++start inferShape 1" << std::endl;
        vector<Shape> result_shape_vec;
        Shape result_shape;
        Tensor tensor_A = inputs[0], tensor_B = inputs[1];
        Shape shape_A = tensor_A->getDims(), shape_B = tensor_B->getDims();
        int len = shape_A.size();
        std::cout << "++++++++++++start inferShape 2" << std::endl;
        if (transA && transB) {
            std::cout << "++++++++++++start inferShape 3" << std::endl;
            Shape new_shape_A;
            for (auto i = 0; i < len - 2; ++i) {
                new_shape_A.emplace_back(shape_A[i]);
            }
            new_shape_A.emplace_back(shape_A[len - 1]);
            new_shape_A.emplace_back(shape_A[len - 2]);

            Shape new_shape_B;
            for (auto i = 0; i < len - 2; ++i) {
                new_shape_B.emplace_back(shape_B[i]);
            }
            new_shape_B.emplace_back(shape_B[len - 1]);
            new_shape_B.emplace_back(shape_B[len - 2]);

            // Get result
            for (auto i = 0; i < len - 2; ++i) {
                result_shape.emplace_back(std::max(new_shape_A[i], new_shape_B[i]));
            }

            result_shape.emplace_back(new_shape_A[len - 2]);
            result_shape.emplace_back(new_shape_B[len - 1]);
            std::cout << "++++++++++++start inferShape 4" << std::endl;
        } else if (transA) {
            std::cout << "++++++++++++start inferShape 5" << std::endl;
            Shape new_shape_A;
            for (auto i = 0; i < len - 2; ++i) {
                new_shape_A.emplace_back(shape_A[i]);
            }
            new_shape_A.emplace_back(shape_A[len - 1]);
            new_shape_A.emplace_back(shape_A[len - 2]);

            // Get result
            for (auto i = 0; i < len - 2; ++i) {
                result_shape.emplace_back(std::max(new_shape_A[i], shape_B[i]));
            }

            result_shape.emplace_back(new_shape_A[len - 2]);
            result_shape.emplace_back(shape_B[len - 1]);
            std::cout << "++++++++++++start inferShape 6" << std::endl;
        } else if (transB) {
            std::cout << "++++++++++++start inferShape 7" << std::endl;
            Shape new_shape_B;
            for (auto i = 0; i < len - 2; ++i) {
                new_shape_B.emplace_back(shape_B[i]);
            }
            new_shape_B.emplace_back(shape_B[len - 1]);
            new_shape_B.emplace_back(shape_B[len - 2]);

            // Get result
            for (auto i = 0; i < len - 2; ++i) {
                result_shape.emplace_back(std::max(shape_A[i], new_shape_B[i]));
            }

            result_shape.emplace_back(shape_A[len - 2]);
            result_shape.emplace_back(new_shape_B[len - 1]);
            std::cout << "++++++++++++start inferShape 8" << std::endl;
        } else {
            std::cout << "++++++++++++start inferShape 9" << std::endl;
            // Get result
            for (auto i = 0; i < len - 2; ++i) {
                result_shape.emplace_back(std::max(shape_A[i], shape_B[i]));
            }

            result_shape.emplace_back(shape_A[len - 2]);
            result_shape.emplace_back(shape_B[len - 1]);
            std::cout << "++++++++++++start inferShape 10" << std::endl;
        }
        result_shape_vec.emplace_back(result_shape);
        return result_shape_vec;
    }

} // namespace infini