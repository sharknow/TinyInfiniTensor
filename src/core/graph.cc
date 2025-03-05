#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        auto last_op = ops[0];
        vector<Operator> ops_pre_to_be_removed;

        for (size_t i = 1; i < ops.size(); ++i) {
            auto cur_op = ops[i];
            if ((last_op->getOpType() == OpType::Transpose) && (cur_op->getOpType() == OpType::Transpose)) {
                if (last_op->getOutput()->getGuid() == (cur_op->getInputs()[0])->getGuid()) {
                    if ((last_op->getInputs()[0])->getDims() == cur_op->getOutput()->getDims()) {
                        ops_pre_to_be_removed.emplace_back(cur_op);
                        auto last_op_input_tensor = last_op->getInputs()[0];
                        auto cur_op_output_tensor = cur_op->getOutput();
                        removeOperator(last_op);
                        removeOperator(cur_op);
                        removeTensor(last_op->getOutput());
                        for (auto &op : ops) {
                            if ((op->getInputs()[0])->getGuid() == cur_op_output_tensor->getGuid()) {
                                op->replaceInput(cur_op_output_tensor, last_op_input_tensor);
                                last_op_input_tensor->removeTarget(last_op);
                                removeTensor(cur_op_output_tensor);
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            last_op = cur_op;
        }

        Tensor new_tensor_A, new_tensor_B, output_tensor_C;
        bool trans_A = false, trans_B = false;
        for (auto &op : ops) {
            if (op->getOpType() == OpType::MatMul) {
                auto input_tensor_A = op->getInputs()[0];
                auto input_tensor_B = op->getInputs()[1];
                output_tensor_C = op->getOutput();

                // Find transpose output tensor_A
                new_tensor_A = input_tensor_A;
                new_tensor_B = input_tensor_B;
                for (auto op_elem : ops) {
                    if (op_elem->getOpType() == OpType::Transpose) {
                        if (op_elem->getOutput()->getGuid() == input_tensor_A->getGuid()) {
                            std::cout << "get new_tensor_A" << std::endl;
                            new_tensor_A = op_elem->getInputs()[0];
                            trans_A = true;
                            op->replaceInput(input_tensor_A, new_tensor_A);
                            new_tensor_A->removeTarget(op_elem);
                        }
                        if (op_elem->getOutput()->getGuid() == input_tensor_B->getGuid()) {
                            new_tensor_B = op_elem->getInputs()[0];
                            std::cout << "get new_tensor_B new Guid = " << new_tensor_B->getGuid() << std::endl;
                            trans_B = true;
                            op->replaceInput(input_tensor_B, new_tensor_B);
                            new_tensor_B->removeTarget(op_elem);
                        }
                        ops_pre_to_be_removed.emplace_back(op_elem);
                        std::cout << "0 now ops size = " << ops.size() << std::endl;
                        removeOperator(op_elem);
                        std::cout << "1 now ops size = " << ops.size() << std::endl;
                        removeTensor(op_elem->getOutput());
                        std::cout << "2 now ops tensor size = " << getTensors().size() << std::endl;
                        break;
                    }
                    std::cout << "3 now ops size = " << ops.size() << std::endl;
                }
                // method: 1
                std::cout << "start remove MatMul op" << std::endl;
                // removeOperator(op);
                std::cout << "4 now ops size = " << ops.size() << std::endl;
                std::cout << "start add new MatMul op, transA = " << trans_A << " transB = " << trans_B << std::endl;
                break;
            }
        }

        for (auto &op : ops) {
            if (op->getOpType() == OpType::MatMul) {
                auto op_pred = op->getPredecessors();
                std::cout << "0 op_pred size = " << op_pred.size() << std::endl;
                for (auto &elem : ops_pre_to_be_removed) {
                    op->removePredecessors(elem);
                }
                std::cout << "1 op_pred size = " << op_pred.size() << std::endl;
                auto matmul_op = as<MatmulObj>(op);
                if (trans_A) {
                    matmul_op->setTransA(true);
                }
                if (trans_B) {
                    matmul_op->setTransB(true);
                }
                break;
            }
        }

        for (auto &op : ops) {
            for (auto &input : op->getInputs()) {
                if (input) {
                    input->addTarget(op);
                    input->setSource(nullptr);
                }
            }

            for (auto &output : op->getOutputs()) {
                if (output) {
                    output->setSource(op);
                }
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        vector<size_t> tensors_offset;
        for (auto &tensor : this->tensors) {
            tensors_offset.emplace_back(this->allocator.alloc(tensor->getBytes()));
        }

        size_t offset_index = 0;
        for (auto &tensor : this->tensors) {
            std::cout << "cur tensor offset = " << tensors_offset[offset_index] << std::endl;
            tensor->setDataBlob(make_ref<BlobObj>(this->runtime, static_cast<uint8_t *>(this->allocator.getPtr()) + tensors_offset[offset_index]));
            ++offset_index;
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini