import torch
import torch.nn as nn
import torch.nn.functional as F
# from ot import sinkhorn  # 使用POT库来计算Sinkhorn距离

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, prob_in, prob_gt, valid):
        loss = self.loss(prob_in, prob_gt) * valid
        return loss

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)

        # prevent NaN loss
        loss[torch.isnan(loss)] = 0
        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid

        # prevent NaN loss
        loss[torch.isnan(loss)] = 0
        return loss

class KLLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output, valid):

        teacher_probs = F.softmax(teacher_output / self.temperature, dim=1)

        with torch.no_grad():
            soft_target_loss = F.kl_div(F.log_softmax(student_output / self.temperature, dim=1), teacher_probs,
                                        reduction='none')
            soft_target_loss = soft_target_loss * valid.float()
            soft_target_loss = soft_target_loss.sum(dim=1)

        soft_target_loss[torch.isnan(soft_target_loss)] = 0

        return soft_target_loss.mean()

class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps
        self.cos = nn.CosineSimilarity(dim=1, eps=eps)

    def forward(self, student_output, teacher_output, valid):

        student_probs = F.softmax(student_output, dim=1)
        teacher_probs = F.softmax(teacher_output, dim=1)


        student_probs = student_probs * valid.float()
        teacher_probs = teacher_probs * valid.float()

        cosine_sim = self.cos(student_probs, teacher_probs)

        loss = 1 - cosine_sim

        loss[torch.isnan(loss)] = 0

        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, student_output, teacher_output, valid):
        student_probs = F.softmax(student_output, dim=1)
        teacher_probs = F.softmax(teacher_output, dim=1)

        student_probs = student_probs * valid.float()
        teacher_probs = teacher_probs * valid.float()

        dice_loss = 0
        for i in range(student_output.shape[1]):
            weight = 1.0 if self.weights is None else self.weights[i]
            intersection = torch.sum(student_probs[:, i] * teacher_probs[:, i])
            student_sum = torch.sum(student_probs[:, i] * student_probs[:, i])
            teacher_sum = torch.sum(teacher_probs[:, i] * teacher_probs[:, i])
            dice = (2. * intersection + self.smooth) / (student_sum + teacher_sum + self.smooth)
            dice_loss += weight * (1 - dice)

        return dice_loss / student_output.shape[1]

class SimilarityPreservingLoss(nn.Module):
    def __init__(self, temperature=1.0, gamma=1.0):
        super(SimilarityPreservingLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma

    def forward(self, student_output, teacher_output):

        student_output = student_output.view(student_output.size(0), -1)
        teacher_output = teacher_output.view(teacher_output.size(0), -1)

        student_sim = F.normalize(student_output @ student_output.t(), p=2, dim=1)
        teacher_sim = F.normalize(teacher_output @ teacher_output.t(), p=2, dim=1)
        distill_loss = (student_sim - teacher_sim).pow(2).mean()

        return distill_loss * self.gamma

class klLoss(nn.Module):
    def __init__(self, T=4.0):
        super(klLoss, self).__init__()
        self.T = T  # 温度系数
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, student_output, teacher_output, valid_mask=None):
        if valid_mask is not None:
            mse_loss = self.mse(student_output * valid_mask, teacher_output * valid_mask)
        else:
            mse_loss = self.mse(student_output, teacher_output)

        s_logits = F.log_softmax(student_output / self.T, dim=-1)
        t_logits = F.softmax(teacher_output / self.T, dim=-1)

        if valid_mask is not None:
            kl_loss = self.kl_div(s_logits * valid_mask, t_logits * valid_mask)
        else:
            kl_loss = self.kl_div(s_logits, t_logits)

        loss = mse_loss + (self.T * self.T) * kl_loss

        return loss