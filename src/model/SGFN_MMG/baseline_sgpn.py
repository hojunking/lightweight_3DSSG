import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_TripletGCN import TripletGCNModel
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti

class SGPN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('SGPN', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_point = 3
        dim_point_rel = 3
        if mconfig.USE_RGB:
            dim_point +=3
            dim_point_rel+=3
        if mconfig.USE_NORMAL:
            dim_point +=3
            dim_point_rel+=3
            
        if mconfig.USE_CONTEXT:
            dim_point_rel += 1

        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim

        dim_point_feature = 256

        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        self.gcn = TripletGCNModel(
            num_layers=self.mconfig.N_LAYERS,
            dim_node=mconfig.point_feature_size,
            dim_edge=mconfig.edge_feature_size,
            dim_hidden=mconfig.point_feature_size,
            use_bn=with_bn)
        
        self.obj_predictor = PointNetCls(num_obj_class, in_size=256,
                                 batch_norm=with_bn, drop_out=True)

        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()


    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)


        rel_feature = self.rel_encoder(edge_feature)

        #print(f'obj_feature: {obj_feature.shape}, rel_feature: {rel_feature.shape}, edge_indices: {edge_indices.shape}')
        obj_feature, rel_feature = self.gcn(obj_feature, rel_feature, edge_indices)

        rel_cls = self.rel_predictor(rel_feature)

        obj_logits = self.obj_predictor(obj_feature)

        return obj_logits, rel_cls
        
    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

        
        loss = 0.1 * loss_obj + loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
    
    
    def kd_process_train(self, teacher_model, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)
        
        
        #obj_logits_3d, obj_logits_2d, rel_cls_3d, rel_cls_2d, obj_feature_3d, obj_feature_2d, edge_feature_2d, obj_logit_scale = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        #t_obj_logits_3d, t_obj_logits_2d, t_rel_cls_3d, t_rel_cls_2d, t_obj_feature_3d, t_obj_feature_2d, t_edge_feature_2d, t_obj_logit_scale = teacher_model(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        t_obj_logits_3d, t_obj_logits_2d, t_rel_cls_3d, t_rel_cls_2d = teacher_model(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        
         # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

        ############ KD ############
        t_loss_obj_3d = F.mse_loss(obj_pred, t_obj_logits_3d)
        t_loss_obj_2d = F.mse_loss(obj_pred, t_obj_logits_2d)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]                
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)

            ############ KD ############
            t_loss_rel_3d = F.binary_cross_entropy(rel_pred,t_rel_cls_3d, weight=weight)
            t_loss_rel_2d = F.binary_cross_entropy(rel_pred,t_rel_cls_2d, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight=weight)

            ############ KD ############
            t_loss_rel_3d = F.binary_cross_entropy(rel_pred, t_rel_cls_3d, weight=weight)
            t_loss_rel_2d = F.binary_cross_entropy(rel_pred, t_rel_cls_2d, weight=weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        
        ############ KD ############
        # t_obj_feature_3d = t_obj_feature_3d / t_obj_feature_3d.norm(dim=-1, keepdim=True)
        # t_obj_feature_2d = t_obj_feature_2d / t_obj_feature_2d.norm(dim=-1, keepdim=True)
        # t_loss_mimic = self.cosine_loss(t_obj_feature_3d, t_obj_feature_2d, t=0.8)

        # compute similarity between visual with text
        rel_text_feat = self.get_rel_emb(gt_cls, gt_rel_cls, edge_indices)

        ############ KD ############
        # t_edge_feature_2d = t_edge_feature_2d / t_edge_feature_2d.norm(dim=-1, keepdim=True)
        # t_rel_mimic_2d = F.l1_loss(t_edge_feature_2d, rel_text_feat)

        loss = lambda_o * loss_obj + 3 * lambda_r * loss_rel
        
        ############ KD ############
        t_alpha = 0.5
        t_loss = lambda_o * t_alpha* (t_loss_obj_2d + t_loss_obj_3d) + 3 * lambda_r * (t_loss_rel_2d + t_loss_rel_3d) #+ 0.1 * (t_loss_mimic + t_rel_mimic_2d)
        loss = loss + t_loss
        

        self.backward(loss)
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
                ("train/T_rel_loss", t_loss_rel_3d.detach().item()),
                ("train/T_obj_loss", t_loss_obj_3d.detach().item()),
                ("train/T_2d_rel_loss", t_loss_rel_2d.detach().item()),
                ("train/T_2d_obj_loss", t_loss_obj_2d.detach().item()),
            ]
        return log
        
           
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
 
        obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
     
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
