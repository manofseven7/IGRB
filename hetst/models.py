"""Relation-aware attention with incoming-neighbor normalization and external self term."""
import torch
from torch import nn
import torch.nn.functional as F

class RelAttention(nn.Module):
    def __init__(self,inp,width,heads,concat=True,relations=2,edge_features=True,self_term=True):
        super().__init__();self.k=heads;self.d=width;self.concat=concat;self.edge_features=edge_features;self.self_term=self_term
        self.w=nn.Parameter(torch.empty(relations,heads,inp,width));self.a=nn.Parameter(torch.empty(relations,heads,3*width))
        self.e=nn.Linear(4,width,bias=False);self.s=nn.Linear(inp,heads*width,bias=False)
        nn.init.xavier_uniform_(self.w);nn.init.xavier_uniform_(self.a)
    def forward(self,x,ef,src,dst,rel,keep=None):
        B,N,_=x.shape; r=rel if len(self.w)>1 else torch.zeros_like(rel)
        u=torch.einsum('bei,ekid->bekd',x[:,src],self.w[r]);v=torch.einsum('bei,ekid->bekd',x[:,dst],self.w[r])
        e=self.e(ef).unsqueeze(2).expand(-1,-1,self.k,-1) if self.edge_features else torch.zeros_like(u)
        score=F.leaky_relu((torch.cat([u,v,e],-1)*self.a[r]).sum(-1),.2)
        mask=dst[:,None]==torch.arange(N,device=x.device)[None,:]
        if keep is not None:mask=mask & keep[:,None]
        # [B, node, head, edge]; incoming softmax, empty neighborhoods have zero messages.
        s=score.permute(0,2,1)[:,None].expand(-1,N,-1,-1)
        m=mask.T[None,:,None,:];s=s.masked_fill(~m,-1e9)
        alpha=torch.softmax(s,-1)*m
        alpha=alpha/(alpha.sum(-1,keepdim=True).clamp_min(1e-12))
        out=torch.einsum('bnke,bekd->bnkd',alpha,u)
        if self.self_term:out=out+self.s(x).reshape(B,N,self.k,self.d)
        out=F.elu(out).flatten(2) if self.concat else F.elu(out.mean(2))
        return out,alpha

class Model(nn.Module):
    def __init__(self,kind,inp=24,width=16,heads=2,layers=3,edge_features=True,self_term=True,temporal=True):
        super().__init__();self.kind=kind;self.temporal=temporal;self.attention=None
        self.g=nn.ModuleList();self.norm=nn.ModuleList(); self.drop=nn.Dropout(.1)
        if kind in ['HetST','HomGAT','RelGRU','NoTime','NoEdge','NoSelf','SingleHead','ClsOnly','RegOnly']:
            if kind=='SingleHead':heads=1
            d=inp
            for l in range(layers):
                out=width*heads if l<layers-1 else width
                self.g.append(RelAttention(d,width,heads,l<layers-1,relations=1 if kind=='HomGAT' else 2,edge_features=kind!='NoEdge',self_term=kind!='NoSelf'))
                self.norm.append(nn.LayerNorm(out));d=out
            if kind=='RelGRU':self.rnn=nn.GRU(width,width,batch_first=True)
        elif kind in ['GCN','RGCN','TGRU']:
            self.proj=nn.ModuleList();self.selfs=nn.ModuleList();d=inp
            for _ in range(layers):
                self.proj.append(nn.ModuleList([nn.Linear(d,width,bias=False) for _ in range(2 if kind=='RGCN' else 1)]));self.selfs.append(nn.Linear(d,width));self.norm.append(nn.LayerNorm(width));d=width
            if kind=='TGRU':self.rnn=nn.GRU(width,width,batch_first=True)
        elif kind=='LSTM':self.rnn=nn.LSTM(inp,width,batch_first=True)
        elif kind=='MLP':self.mlp=nn.Sequential(nn.Linear(inp,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        else:raise ValueError(kind)
        self.cls=nn.Sequential(nn.Linear(width,max(4,width//2)),nn.ReLU(),nn.Linear(max(4,width//2),1))
        self.reg=nn.Sequential(nn.Linear(width,max(4,width//2)),nn.ReLU(),nn.Linear(max(4,width//2),1))
    def forward(self,x,edge,src,dst,rel,keep=None):
        # x: batch, history, node, feature. Snapshot models use last history only.
        B,W,N,D=x.shape
        recurrent=self.kind in ['RelGRU','TGRU','LSTM']
        h=x.reshape(B*W,N,D) if recurrent else x[:,-1]
        ef=edge.reshape(B*W,edge.shape[2],4) if recurrent else edge[:,-1]
        if self.kind=='NoTime':h=torch.cat([h[...,:6],torch.zeros_like(h[...,6:])],-1)
        if len(self.g):
            for layer,norm in zip(self.g,self.norm):h,self.attention=layer(h,ef,src,dst,rel,keep);h=norm(self.drop(h))
        elif self.kind in ['GCN','RGCN','TGRU']:
            for projs,sl,norm in zip(self.proj,self.selfs,self.norm):
                out=sl(h)
                for r,pr in enumerate(projs):
                    mask=(rel==r) if len(projs)>1 else torch.ones_like(rel,dtype=torch.bool)
                    if keep is not None:mask=mask&keep
                    adj=torch.zeros(N,N,device=h.device);adj[dst[mask],src[mask]]=1
                    adj=adj/adj.sum(1,keepdim=True).clamp_min(1)
                    out=out+torch.einsum('nm,bmd->bnd',adj,pr(h))
                h=norm(self.drop(F.elu(out)))
        elif self.kind=='MLP':h=self.mlp(h)
        if recurrent:
            h=h.reshape(B,W,N,-1).permute(0,2,1,3).reshape(B*N,W,-1)
            h=self.rnn(h)[0][:,-1].reshape(B,N,-1)
        return self.cls(h).squeeze(-1),F.softplus(self.reg(h).squeeze(-1))
