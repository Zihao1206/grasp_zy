""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
from models.gqcnn_server import ops
import torch.nn.functional as F

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     # 'skip_connect', # identity
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'std_conv_3x3',
#     'std_conv_5x5',
#     'stw_conv_3x3',
#     'stw_conv_5x5',
#     'none'
# ]

# PRIMITIVES = [
#     # 'skip_connect',
#     'sre_conv_5x3',
#     'std_conv_5x5',
#     'stw_conv_3x3',
#     'stw_conv_5x5',
#     'spl_conv_3x5',
#     'std_conv_3x3'
# ]

PRIMITIVES = [
    'skip_connect',
    # 'sre_conv_5x3',
    # 'spl_conv_3x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'std_conv_5x5',
    'std_conv_3x3',
    # 'stw_conv_3x3',
    # 'stw_conv_5x5'
]

# 卷积层浮点数运算公式，conv flops = Cout * Cin * k * k * Hout * Wout * BatchSize
# 其中k为卷积核大小，C表示通道数，H和W表示feature map的尺寸
# std_conv_3x3 flops = 16 * 16 * 3 * 3 * 150 * 150 * 4
# FLOPs_dict = {
#     # 'skip_connect' : 0,
#     'std_conv_3x3' : 207360000,
#     'std_conv_5x5' : 576000000,
#     'stw_conv_3x3' : 414720000,
#     'stw_conv_5x5' : 1152000000,
#     'spl_conv_3x5' : 783360000,
#     'sre_conv_5x3' : 783360000
# }

# sep卷积层浮点数运算公式，sep_conv flops = 2*BatchSize* (Cin * k * k * Hout * Wout + 1 * 1 * Cin * Cout * Hout * Wout)
# dilated卷积层浮点数运算公式，dil_conv flops = BatchSize* (Cin * k * k * Hout * Wout + 1 * 1 * Cin * Cout * Hout * Wout)
# 其中k为卷积核大小，C表示通道数，H和W表示feature map的尺寸
FLOPs_dict = {
    # 'skip_connect' : 0,
    'dil_conv_3x3' : 36000000,
    'dil_conv_5x5' : 59040000,
    'sep_conv_3x3' : 72000000,
    'sep_conv_5x5' : 118080000,
    'spl_conv_3x5' : 783360000,
    'sre_conv_5x3' : 783360000,
    'std_conv_3x3' : 207360000,
    'std_conv_5x5' : 576000000,
    'stw_conv_3x3' : 414720000,
    'stw_conv_5x5' : 1152000000
}

def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, affine = True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag

def to_op_dag(C_in, gene, reduction):
    """ generate numrous discrete ops between nearby nodes from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx, alpha in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 1
            op = ops.OPS[op_name](C_in, stride, affine = True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            op.alpha = alpha
            row.append(op)
        dag.append(row)
    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    # assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)
    return gene

def parse_alpha(alpha, k):
    '''
    return k top opearions with their alphas as gene
    gene is list:
    [
        [('node1_ops_1', node_idx, alpha), ..., ('node1_ops_k', node_idx, alpha)],
        [('node2_ops_1', node_idx, alpha), ..., ('node2_ops_k', node_idx, alpha)],
        ...
    ]
    '''
    gene = []
    s = 0
    for edges in alpha:
        edge_max, primitive_indics = torch.topk(edges[:,:],k)
        weights_normal = [F.softmax(edge_max[0], dim=-1) ]
        # topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for operation_idx, alpha in zip(primitive_indics[0],weights_normal[0]):
            prim = PRIMITIVES[operation_idx.item()]
            # node_gene.append((prim, s, alpha.item()))
            node_gene.append((prim, s, alpha.item()))
        gene.append(node_gene)
        s = s+1
    return gene        

def floating_point(alpha, k):
    '''
    return k top opearions with their FLOPs
    '''
    float_num = 0
    max_float = k * max(FLOPs_dict.values())


    for sm_alpha in alpha:
        for i,a in enumerate(sm_alpha[0]):
            prim = PRIMITIVES[i]
            float_num = float_num + a * FLOPs_dict[prim]


    return float_num / max_float