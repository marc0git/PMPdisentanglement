from argparse import ArgumentParser
from distutils.util import strtobool
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import random

from torch import nn
from torch.utils.data import DataLoader
# from dataset import *
from utils import get_env,load_envs,tv_loss

import numpy as np

from dataset import *



dataset_names = ['dsprites_full','shapes3d','cars3d','small-norb','faust','dsprites_aggr']
    
class Model(pl.LightningModule):

    def __init__(self, hparams, encoder=None, encoders=None, decoder=None):
        load_envs()
        super(Model, self).__init__()
        # assert hparams.latentdim == hparams.latentdim1 + hparams.latentdim2,"Inconsistent dimensions for the latent space!"
        self.hparams = hparams
        self.lr= self.hparams.learning_rate
        torch.set_default_tensor_type(torch.FloatTensor)
        self.latentdim = self.hparams.latentdim
        self.n_latent_factors = self.hparams.n_latent_factors
        
        self.encoders = encoders
        self.encoder = encoder
        self.decoder=decoder

        self.criterion= nn.BCELoss(reduction="mean") if self.hparams.n_dataset == 0 else nn.MSELoss(reduction="mean")
        
        self.aggregator = None  

        self.eye_ld=None
        
        if self.encoders is None:
            self.encoders = nn.ModuleList()
            for i in range(self.n_latent_factors):
                self.encoders.append(nn.Sequential(
                    nn.Linear(self.hparams.latentdim, self.hparams.latentdim),
                    nn.ReLU(True),
                    nn.Linear(self.hparams.latentdim, self.hparams.latentdim),
                    # nn.LayerNorm(self.hparams.latentdim)
                ))
        

    def pre_process(self,x):
        if len(x.shape)==3:
            x=x[None,...]
        x = x.permute(0,3,1,2)
        return x
    
    
    def post_process(self, x: torch.Tensor):
        x = self.squash(x.permute(0,2,3,1))
        return x

    def aggregate(self, x):
        if self.aggregator is not None:
            x = self.aggregator(torch.cat(x, -1))
            return x
        else:
            if type(x) is list or type(x) is tuple:
                x = torch.stack(x)
            return x.sum(0)

    def latent_encode(self, x):
        z_latents1 = []
        for i in range(self.n_latent_factors):
            tmp = self.encoders[i](x)
            z_latents1.append(tmp)
        return z_latents1

    def squash(self, x):
            return torch.sigmoid(x) if self.hparams.n_dataset==0 else x

    def forward(self, xx: torch.Tensor):
        xx = self.pre_process(xx)
        zz = self.encoder(xx)
        
        z_latents = self.latent_encode(zz)
        z_aggrs = self.aggregate(z_latents)
        
        outs = self.decoder(z_aggrs)
        outs = self.post_process(outs)
        
        return {"yy": outs,
                "z_latents": z_latents,
                "zz": zz,
                "z_aggrs": z_aggrs
                }


    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)

        
    def training_step(self, batch, batch_idx):
        
        epochs = self.hparams.max_epochs
        
        loss_cons_reg = torch.tensor(0)
        losses_cons = torch.tensor(0)
        loss_sparse = torch.tensor(0)
        loss_dist = torch.tensor(0)
        
        
        x1, x2, _ = batch
        xx = torch.cat([x1,x2],0)
        outputs = self.forward(xx)
        
        yy, z_latents, zz, z_aggr = \
                    outputs["yy"],\
                    outputs["z_latents"], \
                    outputs["zz"], \
                    outputs["z_aggrs"]
        
        #split in z1 and z2 in the first dimension
        z_latents = torch.stack(z_latents,0)\
              .view(len(z_latents),2,-1,z_latents[0].shape[-1])\
              .transpose(0,1)
        zz = zz.view( *([2,-1] + list(zz.shape[1:])) )
        z_aggr = z_aggr.view( *([2,-1] + list(z_aggr.shape[1:])) )

        
        beta1 = min(np.exp((25/epochs)*(self.current_epoch - epochs*0.4)), 1)
        beta2 = min(np.exp((25/epochs)*(self.current_epoch - epochs*0.5)), 1)
        beta3 = 1-min(np.exp((25/epochs)*(self.current_epoch - epochs*0.4)), 1)
       
    
        ############################### reconstruction loss
        loss_rec = self.criterion(yy,xx)
        

        #Start only with reconstruction loss for the first 25% of epochs
        if self.current_epoch>epochs*0.25:
            
            ################################ SPARSE LOSS
            if self.eye_ld is None:
                self.eye_ld = torch.eye(z_latents.shape[1]).to(z_latents.device)
            all_z = z_latents.permute(0,2,3,1)
            loss_sparse = (all_z@(1-self.eye_ld)*all_z).abs().sum(-1).mean()

            ################################ consistency loss
            if beta2>1e-2 and self.hparams.lambda_consistency>0:
                _, nl, bs, d = z_latents.shape

                z_misc = []
                z_latents_miscs_pre = []
                for i in range(self.n_latent_factors):
                    l = z_latents[1,:,:,:]+0            
                    l[i] = z_latents[0,i,:,:]+0
                    z_misc.append(self.aggregate(l))
                    z_latents_miscs_pre.append(l)
                z_latents_miscs_pre = torch.stack(z_latents_miscs_pre,1).view(nl,nl*bs,d)

                #subsample bs*2 combinations
                idxs = torch.randperm(z_latents_miscs_pre.shape[1])[:bs*2]
                z_latents_miscs_pre =  z_latents_miscs_pre[:,idxs,:]

                z_miscs = torch.stack(z_misc,0).view(-1,d)[idxs] #nl,bs,d -> nlxbs,d

                decoded = self.decoder(z_miscs)#.detach()
                out_misc = self.post_process(decoded)
                z_latents_miscs = self.latent_encode(self.encoder(self.squash(decoded)))
                z_latents_miscs = torch.stack(z_latents_miscs,0)

                losses_cons = F.mse_loss(z_latents_miscs,z_latents_miscs_pre,reduction="mean")

            ################################## CONS REG + DIST
            _, nl, bs, d = z_latents.shape

            dist = (z_latents[0]-z_latents[1]).pow(2).sum(-1).t()
            zs = z_latents.transpose(0,1).view(nl,2*bs,d)
            mean_dist = torch.cdist(zs,zs).mean([-2,-1])+1e-8

            dist_norm = dist/mean_dist
            mask = 1-torch.softmax(1e6*dist_norm,-1).detach()
            
            loss_cons_reg = (dist*mask).mean() + 10*(0.05-dist*(1-mask)).relu().mean()
            
            ###### distribution loss #######
            st = mask.t().detach()
            sel = torch.softmax(1e3*dist,-1)
            loss_dist = 1e2*(1/sel.shape[-1] - sel.mean(0)).pow(2).mean()

            ##############################################

       
        loss = loss_rec  \
               + beta1 * (self.hparams.lambda_distance*loss_cons_reg +\
                          self.hparams.lambda_sparsity*loss_sparse) \
               + beta2 *  self.hparams.lambda_consistency * losses_cons \
               + beta3 *  self.hparams.lambda_distribution * loss_dist
        
#         if batch_idx%8==0:
        self.log("train_loss", loss.item())
        self.log("loss_rec", loss_rec.item())
        self.log("loss_sparse", loss_sparse.item())
        self.log("loss_consistency", losses_cons.item())
        self.log("loss_distance", loss_cons_reg.item())
        self.log("loss_distrib", loss_dist.item())
        self.log("beta1", beta1)
        self.log("beta2", beta2)
        self.log("beta3", beta3)

        if batch_idx==0:
            self.logger.experiment.log({"trainx1": [wandb.Image((xx[0, ...]).cpu().numpy(), caption=" trainx1"),
            wandb.Image(torch.clip(yy[0, ...],0,1).detach().cpu().numpy(), caption="trainy1")]})

        return {"loss": loss}#, loss_rec



    def validation_step(self, batch, batch_idx):
        x1, x2,idt = batch
        xx = torch.cat([x1,x2],0)
        
        outputs = self.forward(xx)
        
        yy, z_latents, zz, z_aggr = \
                    outputs["yy"],\
                    outputs["z_latents"], \
                    outputs["zz"], \
                    outputs["z_aggrs"]
        
        #reconstruction loss
        loss_rec = self.criterion(xx,yy)
        
        #split in z1 and z2 in the first dimension
        yy = yy.view( *([2,-1] + list(yy.shape[1:])) )
        z_latents = torch.stack(z_latents,0)\
              .view(len(z_latents),2,-1,z_latents[0].shape[-1])\
              .transpose(0,1)
        zz = zz.view( *([2,-1] + list(zz.shape[1:])) )
        z_aggr = z_aggr.view( *([2,-1] + list(z_aggr.shape[1:])) )
        
        
        self.log("val_loss", loss_rec.detach().cpu())
        return {
            "val_loss": loss_rec.detach().cpu()
        }

    def validation_epoch_end(self, outputs):
        # LOGS
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # tensorboard = self.logger[1].experiment
        idx1 = random.randint(0, len(self.val_set) - 1)
        idx2 = random.randint(0, len(self.val_set) - 1)
        
        x1 = self.val_set[idx1][0].cuda()
        x2 = self.val_set[idx2][0].cuda()
        
        I = self.get_dis_image(torch.stack([x1,x2]))
        self.logger.experiment.log({'dis': [
            wandb.Image(I.cpu().numpy(), caption="dis")]})

        
    def get_dis_image(self, xx):
        model=self
        img1 = xx[0]
        img2 = xx[1]
        col = img1[:,:5,:]*0+1

        outputs = model.forward(torch.cat([img1[None,...], img2[None,...]],0))

        rec1 = outputs['yy'][0].detach().clip(0,1)
        rec2 = outputs['yy'][1].detach().clip(0,1)

        I=None
        for i in range(len(outputs["z_latents"])):
            z1 = [x[0]+0 for x in outputs["z_latents"]]
            z2 = [x[1]+0 for x in outputs["z_latents"]]
            a = torch.linspace(1,0,4)[1:,None].to(xx.device)  
            z1 = [z1.repeat(a.shape[0],1) for z1,z2 in zip(z1,z2)]
            z1[i] = z1[i][0,:]*a + (1-a)*z2[i][:]

            z = model.aggregate(z1)
            res = model.post_process(model.decoder.forward(z).detach()).clip(0,1)  
            res = res.flatten().reshape(*list(res.shape))
            Irow = torch.cat([img1,col,col ,rec1, col]+sum([[r,col] for r in res],[])+ [rec2,col,col,img2],1)
            if I is None:
                I=Irow
            else:
                I = torch.cat([I,Irow],0)
        return I


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.hparams.weight_decay)

    def prepare_data(self) -> None:
        print('dataset_loading...')
        
        dataset_name = dataset_names[self.hparams.n_dataset]

        if dataset_name == 'dsprites_aggr':
            print("dsprites_aggr")
            self.train_set = DatasetVariableK(dataset_name='dsprites_aggr',k=self.hparams.k, factors=[1,2,4])
            self.val_set= self.train_set
        else:            
            print("%s with variable k" % dataset_name)
            self.train_set = DatasetVariableK(dataset_name = dataset_name, factors=None, k=self.hparams.k)
            self.val_set = self.train_set
        print('Done')

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=strtobool(self.hparams.train_shuffle),
            num_workers=32,
            drop_last=True
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=32,
            drop_last=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--n_latent_factors", default=10, type=int)
        parser.add_argument("--latentdim", default=10, type=int)

        parser.add_argument("--n_dataset", default=0, type=int)
        parser.add_argument("--k", default=1, type=int)

        parser.add_argument("--lambda_distribution", default=1e-4, type=float)
        parser.add_argument("--lambda_sparsity", default=1e-1, type=float)
        parser.add_argument("--lambda_consistency", default=1e2, type=float)
        parser.add_argument("--lambda_distance", default=1e-1, type=float)

        
        # training specific (for this model)
        parser.add_argument("--learning_rate", default=5e-4, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--weight_decay", default=1e-6, type=float)
        parser.add_argument("--train_shuffle", default="True", type=str)
        return parser