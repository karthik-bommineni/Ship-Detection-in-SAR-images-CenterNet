import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

'''
在添加gwd loss中使用了优化：
即由于CenterNet中只有唯一的正样本，故预测值和gt的hm的中心是一致的。
故可以直接将gt和pred的中心任意赋值为相同的值即可。
而不会影响到gwd_loss的计算。
    
    problem1: 在自己虚拟128个[1,1]圆心后，更新第二步时候 loss会变成nan。
    分析原因肯能是 不能虚拟1，虚拟个0尝试依旧不行。 
    应该就是虚拟圆心的问题，
    
'''

def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S

'''
loss_weight: 一般设置为 5-10
tau 是一个可以调节的参数： [1.0,2.0,3.0,5.0]
alpha暂定为1.0
'''
def gwd_loss(pred, target, fun='sqrt', tau=1.0, alpha=1.0, normalize=False):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z
    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2
    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)
    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)
    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """

    pred_ = pred[:, 4]/180 * np.pi
    target_ = target[:, 4]/180 * np.pi
    pred = torch.cat((pred[:, :4], pred_[:, None]), dim=-1)
    target = torch.cat((target[:, :4], target_[:, None]), dim=-1)

    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0)
    # distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    else:
        raise ValueError('Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, pred_tensor, target_tensor):
        return self.neg_loss(pred_tensor, target_tensor)

# 根据ind获取feat上位置对应元素
def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()        # [b,h,w,c]
    feat = feat.view(feat.size(0), -1, feat.size(3))    # [b,h*w,c]
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, mask, ind, target):
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)                                   # @@@@@@@@@@@@@每个目标的平均损失
        return loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _relu(x):
    y = torch.clamp(x.relu_(), min=0., max=179.99)
    return y

# 获取gwd loss的标准格式
def format_gt_pred(pred_ang, pred_hw, mask, ind, target_ang, target_hw, target_cxcy): # [2,128,2]
    batch_objs = mask.sum()
    batch = mask.shape[0]
    per_img_objs = torch.count_nonzero(mask, dim=-1)             # [b,]
    # gather ang
    pred_ang = _transpose_and_gather_feat(pred_ang, ind)
    mask_ang = mask.unsqueeze(2).expand_as(pred_ang).float()
    pred_ang = pred_ang * mask_ang                               # [b,128,1]
    # gather wh
    pred_hw = _transpose_and_gather_feat(pred_hw, ind)
    mask_hw = mask.unsqueeze(2).expand_as(pred_hw).float()
    pred_hw = pred_hw * mask_hw                                  # [b,128,2]
    # 创建一个假的中心，此处可能存在错误，导致loss == nan.
    cxcy = target_cxcy
    # 组合pred
    pred_cxcy_hw_ang = torch.cat((cxcy, pred_hw, pred_ang), dim=-1)   # [b,128,5]
    # 组合gt
    gt_cxcy_hw_ang = torch.cat((cxcy, target_hw, target_ang), dim=-1) # [b,128,5]
    # 得到mask掩码
    mask = mask.unsqueeze(2).expand_as(pred_cxcy_hw_ang).float()
    p = (pred_cxcy_hw_ang * mask)                         # [b,128,5] --> [b*128,5]
    g = (gt_cxcy_hw_ang * mask)                           # [b,128,5] --> [b*128,5]
    # 拾取p
    p_ = torch.empty((batch_objs,5), device= torch.device('cuda'))
    for id in range(batch):
        cur_per_img_objs = per_img_objs[id]
        if id ==0:
            p_[:cur_per_img_objs, :] = p[id][0:cur_per_img_objs, :]
        else:
            p_[per_img_objs[id-1]:cur_per_img_objs+per_img_objs[id-1], :] = p[id][0:cur_per_img_objs, :]
    # 拾取g
    g_ = torch.empty((batch_objs,5), device= torch.device('cuda'))
    for id in range(batch):
        cur_per_img_objs = per_img_objs[id]
        if id ==0:
            g_[:cur_per_img_objs, :] = g[id][0:cur_per_img_objs, :]
        else:
            g_[per_img_objs[id-1]:cur_per_img_objs+per_img_objs[id-1], :] = g[id][0:cur_per_img_objs, :]
    #
    # 由于只需计算对应的pred和gt之间的gwd loss，虽然128个中大部分为0
    # 但由于[cx,cy,0,0,0]的pred和gt的gwd loss=0，故不影响最终的gwd loss
    # 将二者转成gwd loss接收的格式： [b,128,5] --> [b*128,5]
    loss = gwd_loss(p_,g_)
    loss = loss.sum() / (batch_objs + 1e-4)
    return loss

class CtdetGWDLoss(torch.nn.Module):
    # loss_weight={'hm_weight':1,'wh_weight':0.1,'ang_weight':0.1,'reg_weight':0.1, gwd_weight:1}
    def __init__(self, loss_weight):
        super(CtdetGWDLoss, self).__init__()
        self.crit = FocalLoss()                 # 类别损失 ok
        self.crit_reg = RegL1Loss()             # 中心偏移损失 ok
        self.crit_wh = RegL1Loss()    # ang 和 hw 的损失保留，便于后续损失优化
        self.loss_weight = loss_weight

    def forward(self, pred_tensor, target_tensor):
        hm_weight = self.loss_weight['hm_weight']
        wh_weight = self.loss_weight['wh_weight']
        reg_weight = self.loss_weight['reg_weight']
        ang_weight = self.loss_weight['ang_weight']
        gwd_weight = self.loss_weight['gwd_weight']

        hm_loss, wh_loss, off_loss, ang_loss, total_gwd_loss = 0, 0, 0, 0, 0
        pred_tensor['hm'] = _sigmoid(pred_tensor['hm'])
        #        print(target_tensor['hm'].size())
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm'])  # hm_loss: ok

        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg'], target_tensor['reg_mask'], target_tensor['ind'],
                                      target_tensor['reg'])
        # 这两项的L1损失先不添加。
        if ang_weight > 0:
            pred_tensor['ang'] = _relu(pred_tensor['ang'])
            ang_loss += self.crit_wh(pred_tensor['ang'], target_tensor['reg_mask'], target_tensor['ind'],
                                     target_tensor['ang'])
        if wh_weight > 0:
            wh_loss += self.crit_wh(pred_tensor['wh'], target_tensor['reg_mask'], target_tensor['ind'],
                                    target_tensor['wh'])
        # gwd_loss
        total_gwd_loss += format_gt_pred(pred_tensor['ang'], pred_tensor['wh'], target_tensor['reg_mask'], target_tensor['ind'], target_tensor['ang'], target_tensor['wh'], target_tensor['cxcy'])
        return hm_weight * hm_loss + wh_weight * wh_loss + reg_weight * off_loss + ang_weight * ang_loss + gwd_weight * total_gwd_loss
        #return hm_weight * hm_loss + reg_weight * off_loss  + gwd_weight * total_gwd_loss

if __name__ == '__main__':
    # hm: torch.Size([2, 20, 128, 128])
    # wh: torch.Size([2, 2, 128, 128])
    # ang: torch.Size([2, 1, 128, 128])
    # reg: torch.Size([2, 2, 128, 128])
    # input: torch.Size([2, 3, 512, 512])
    # hm: torch.Size([2, 20, 128, 128])
    # reg_mask: torch.Size([2, 128])
    # ind: torch.Size([2, 128])
    # wh: torch.Size([2, 128, 2])
    # ang: torch.Size([2, 128, 1])
    # reg: torch.Size([2, 128, 2])

    print('测试数据1：')
    pred1 = torch.FloatTensor([[70, 70, 70, 10, -30 ],
                              [50, 50, 100, 700, -30]])
    target1 = torch.FloatTensor([[70, 70, 70, 10, 150 ],
                                [10, 40, 100, 700, -40 ]])

    #print(gwd_loss(pred1,target1))

