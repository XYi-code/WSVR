import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np

class MemoryLayer(Function):
    def __init__(self, memory):
        super(MemoryLayer, self).__init__()
        self.memory = memory
        self.global_norm = nn.BatchNorm1d(2048)

    @staticmethod
    def forward(ctx, inputs, targets, memory):
        ctx.save_for_backward(inputs, targets)
        ctx.memory = memory
        outputs = inputs.mm(memory.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        memory = ctx.memory
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(memory)
        for x, y in zip(inputs, targets):
            memory[y] = 0.5 * memory[y] + 0.5 * x
            memory[y] /= memory[y].norm()
        return grad_inputs, None, None

class Memory(nn.Module):
    def __init__(self, num_features, num_classes, num_cam, alpha=0.01):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_cam = num_cam
        self.alpha = alpha
        self.global_norm = nn.BatchNorm1d(num_features)
        self.register_buffer('mem', torch.zeros(num_classes, num_features))
        self.register_buffer('mem_cam', torch.zeros(num_cam, num_features))
        self.register_buffer('mem_TID', torch.empty(num_classes, dtype=torch.long))
        self.register_buffer('mem_CID', torch.empty(num_classes, dtype=torch.long))

    def store(self, inputs, camid, tid, target):
        self.mem_TID[target] = tid.to(self.mem_TID.device)
        self.mem_CID[target] = camid.to(self.mem_CID.device)
        self.mem[target] = inputs.to(self.mem.device)

    def set_cam_memory(self):
        valid_cam_num = 0
        valid_cam_id = []
        valid_mask = self.mem.pow(2).sum(dim=1) > 0
        for i in range(self.num_cam):
            tmp_set = self.mem[(self.mem_CID == i) & valid_mask]
            if tmp_set.size(0) == 0:
                continue
            else:
                valid_cam_id.append(i)
                valid_cam_num += 1
                self.mem_cam[i] = F.normalize(torch.mean(tmp_set, 0), p=2, dim=0)
        return valid_cam_num, valid_cam_id

    def get_cam_likelihood(self, inputs):
        return inputs.mm(self.mem_cam.t())

    def get_cam_mem(self, camid):
        return self.mem[self.mem_CID == camid], self.mem_TID[self.mem_CID == camid]

    def forward(self, inputs, targets, epoch=None):
        inputs = inputs.to(self.mem.device)
        targets = targets.to(self.mem.device)
        logits = MemoryLayer.apply(inputs, targets, self.mem)
        return logits
