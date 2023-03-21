import torch

def get_plane(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    ATWA_1 = torch.inverse(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    # return dn
    dn_up = torch.cat([X[0] * X[2], X[1] * X[2], -X[2]], dim=0),
    dn_norm = X[0] * X[0] + X[1] * X[1] + 1.0
    dn = dn_up[0] / dn_norm

    normal_n = dn / torch.norm(dn)
    for_p2plane = X[2] / torch.sqrt(dn_norm)
    return normal_n, dn, for_p2plane

def get_plane_in_batch(pc, pc_w):
    # pc = torch.rand(2, 3, 5, 3).double()
    # pc_w = torch.rand(2, 3, 5).double()
    A = torch.cat([pc[...,:2], torch.ones_like(pc[...,0].unsqueeze(-1))], dim=-1)
    b = pc[..., 2].unsqueeze(-1)
    W = torch.diag_embed(pc_w, offset=0, dim1=-2, dim2=-1)
    WA = torch.matmul(W, A)
    ATWA = torch.matmul(A.transpose(-1,-2), WA)
    ATWA_1 = torch.inverse(ATWA)

    Wb = torch.matmul(W, b)
    ATWb = torch.matmul(A.transpose(-1,-2), Wb)
    X = torch.matmul(ATWA_1, ATWb)
    dn_up = torch.cat([X[..., 0,:]*X[..., 2,:], X[..., 1,:]*X[..., 2,:], -X[..., 2,:]], dim=-1)
    dn_norm = X[..., 0, :] * X[..., 0, :] + X[..., 1, :] * X[..., 1, :] + 1.0 # 2x1
    dn = dn_up/(dn_norm + 1e-8)
    normal_n = dn/torch.norm(dn, dim=-1, keepdim=True)
    for_p2plane = X[..., 2, :] / torch.sqrt(dn_norm)
    # for i in range(2):
    #     for j in range(3):
    #         print(i, j)
    #         ori_n, ori_dn, ori_p2plane = get_plane_in_one(pc[i,j], pc_w[i,j])
    #         assert  torch.allclose(ori_n, normal_n[i,j])
    #         assert torch.allclose(ori_dn, dn[i, j]), torch.max(torch.abs(ori_dn-dn[i,j]))
    #         assert torch.allclose(ori_p2plane, for_p2plane[i, j])
    return normal_n, dn, for_p2plane


def get_plane_parameter(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    ATWA_1 = torch.inverse(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    return X