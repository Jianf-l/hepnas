import torch
from torch import nn


class DMLLoss(nn.Module):

    def __init__(self):
        super(DMLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean').cuda()

    def forward(self, self_output: torch.Tensor, other_outputs: list[torch.Tensor]):
        length = len(other_outputs)
        loss = 0.0
        for other_output in other_outputs:
            loss += self.criterion(self_output.log_softmax(dim=-1), other_output.softmax(dim=-1))
        return loss / length


class KDLoss(nn.Module):

    def __init__(self):
        super(KDLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean').cuda()

    def forward(self, self_output: torch.Tensor, teacher_output: torch.Tensor):
        return self.criterion(self_output.log_softmax(dim=-1), teacher_output.softmax(dim=-1))


class MultiTeacherLoss(nn.Module):

    def __init__(self):
        super(MultiTeacherLoss, self).__init__()
        self.dml_loss = DMLLoss()
        self.kd_loss = KDLoss()

    def forward(self, self_output: torch.Tensor, teacher_output: torch.Tensor, other_outputs: list[torch.Tensor]):
        dml_loss_ = self.dml_loss(self_output, other_outputs)
        kd_loss_ = self.kd_loss(self_output, teacher_output)
        # print(f"dml loss: {dml_loss_}")
        # print(f"kd loss: {kd_loss_}")
        return dml_loss_ + kd_loss_


class Hint(nn.Module):

    def __init__(self):
        super(Hint, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean').cuda()

    def forward(self, fm_s, fm_t):
        # fm_s: feature map from student
        return self.criterion(fm_s, fm_t)
