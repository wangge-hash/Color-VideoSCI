import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from cacti.datasets.builder import build_dataset 
from cacti.models.builder import build_model
from cacti.utils.optim_builder import  build_optimizer
from cacti.utils.loss_builder import build_loss
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks, generate_masks_real
from cacti.utils.config import Config
from cacti.utils.logger import Logger
from cacti.utils.utils import save_image, load_checkpoints, get_device_info
from cacti.utils.eval import eval_psnr_ssim
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import time
import argparse 
import json 
import einops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--local_rank",default=-1)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg.in_ch
        color_ch = cfg.color_ch
        color_ch_G = color_ch[0] #gray
        color_ch_C = color_ch[1] #color
        cfg0 = {k: (v[0] if k == 'color_ch' else v) for k, v in cfg.items()}
        cfg1 = {k: (v[1] if k == 'color_ch' else v) for k, v in cfg.items()}

        self.model0 = build_model(cfg0)
        self.model1 = build_model(cfg1)

        #(1,1,8,128,128)-->(1,3,8,128,128)
        self.up_conv_G2C = nn.Sequential(
            nn.Linear(color_ch_G, 3),
            nn.LeakyReLU(),
            nn.Linear(3, 3),
            nn.LeakyReLU(),
            nn.Linear(3, color_ch_C),
            nn.LeakyReLU(),
        )

        #(1,6,8,128,128)-->(1,3,8,128,128)
        self.output_fuse = nn.Sequential(
            nn.Conv3d(2*color_ch_C, 3, kernel_size=(3,7,7), stride=1,padding=(1,3,3)),
            nn.LeakyReLU(),
            nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(3, color_ch_C, kernel_size=3, stride=(1,1,1), padding=1),
            nn.LeakyReLU(),
        )
        #(1,256,8,128,128)-->(1,256,8,128,128)
        self.fem0 = nn.Sequential(
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=1, stride=1),
            nn.ELU(),
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.fem1 = nn.Sequential(
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=1, stride=1),
            nn.ELU(),
            nn.Conv3d(in_ch*4, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        #(1,256*3,8,128,128)-->(1,1,8,128,128)
        self.up_conv = nn.Conv3d(in_ch*4*2,in_ch*8*2,1,1)
        self.up = nn.PixelShuffle(2)
        self.vrm = nn.Sequential(
            nn.Conv3d(in_ch*4, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv3d(in_ch*2, in_ch, kernel_size=1, stride=1),
            nn.ELU(),
            nn.Conv3d(in_ch, color_ch_C, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,y_list,Phi_list,Phi_s_list):
        out_list = []
        y0,y1 = y_list
        Phi0,Phi1 = Phi_list
        Phi0_s,Phi1_s = Phi_s_list

        output00,output01 = self.model0(y0,Phi0,Phi0_s)
        output10,output11 = self.model1(y1,Phi1,Phi1_s)

        output00 = self.fem0(output00)
        output10 = self.fem1(output10)

        output0 = torch.cat((output00,output10),dim=1)       
        output0 = self.up_conv(output0)
        output0 = einops.rearrange(output0,"b c t h w-> b t c h w")
        output0 = self.up(output0)
        output0 = einops.rearrange(output0,"b t c h w-> b c t h w")
        output0 = self.vrm(output0)

        # if self.color_ch == 3:
        #     output1 = torch.cat((output01,output11,output21),dim=0)
        #     output1 = einops.rearrange(output1,"b c t h w-> c b t h w")
        #     output1 = self.output_fuse(output1)
        #     output1 = einops.rearrange(output1,"c b t h w-> b c t h w")
        # else:
        # (1,1,8,128,128) --> (1,8,128,128,1) --> (1,8,128,128,3) --> (1,3,8,128,128)
        output01 = einops.rearrange(output01,"b c t h w-> b t h w c")
        output01 = self.up_conv_G2C(output01)
        output01 = einops.rearrange(output01,"b t h w c-> b c t h w")

        output1 = torch.cat((output01,output11),dim=1)
        output1 = self.output_fuse(output1)

        out = output0 + output1

        # if self.color_ch!=3:
        #     out = out.squeeze(1)
        out_list.append(out)
        return output11, out_list


def main(args, cfg):
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])

    if args.resume is not None:
        cfg.resume = args.resume

    log_dir = osp.join(args.work_dir,"log")
    show_dir = osp.join(args.work_dir,"show")
    train_image_save_dir = osp.join(args.work_dir,"train_images")
    checkpoints_dir = osp.join(args.work_dir,"checkpoints")

    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    if not osp.exists(train_image_save_dir):
        os.makedirs(train_image_save_dir)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir = show_dir)

    rank = 0 
    if args.distributed:
        local_rank = int(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    
    device = args.device
    model = Model(cfg.model).to(device)

    if rank==0:
        logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
        logger.info('cfg info:\n'
                + dash_line + 
                json.dumps(cfg, indent=4)+'\n'+
                dash_line) 
        logger.info('Model info:\n'
                + dash_line + 
                str(model)+'\n'+
                dash_line)

    mask0,mask0_s,mask1,mask1_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
    # mask0,mask0_s,mask1,mask1_s,mask2,mask2_s = generate_masks_real(mask_path0='./test_datasets/real_data/left_mask.mat',mask_path1='./test_datasets/real_data/right_mask.mat',mask_shape=(8,128,128))
    train_data = build_dataset(cfg.train_data,{"mask":(mask0,mask1)})
    # if cfg.eval.flag:
    #     test_data = build_dataset(cfg.test_data,{"mask":(mask0,mask1)})
    if args.distributed:
        dist_sampler = DistributedSampler(train_data,shuffle=True)
        train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        sampler=dist_sampler,
                                        num_workers = cfg.data.workers_per_gpu)
    else:
        train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True,
                                        num_workers = cfg.data.workers_per_gpu)
        
    # for param in model.model0.parameters():
    #     param.requires_grad = False
    # for param in model.model1.parameters():
    #     param.requires_grad = False
    # for param in model.model2.parameters():
    #     param.requires_grad = False
    # optimizer = build_optimizer(cfg.optimizer,{"params":filter(lambda p: p.requires_grad, model.parameters())})

    optimizer = build_optimizer(cfg.optimizer,{"params":model.parameters()})
    
    criterion = build_loss(cfg.loss)
    criterion = criterion.to(args.device)
    
    start_epoch = 0
    if rank==0:

        if isinstance(cfg.checkpoints, list):
            logger.info("Load pre_train model...")
            resume_dict0 = torch.load(cfg.checkpoints[0])
            resume_dict1 = torch.load(cfg.checkpoints[1])
            resume_list = [resume_dict0,resume_dict1]
            model_list = [model.model0,model.model1]
            for iii, resume_dict in enumerate(resume_list):
                if "model_state_dict" not in resume_dict.keys():
                    model_state_dict = resume_dict
                else:
                    model_state_dict = resume_dict["model_state_dict"]
                load_checkpoints(model_list[iii],model_state_dict)
        elif cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            resume_dict = torch.load(cfg.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
        else:            
            logger.info("No pre_train model")

        if cfg.resume is not None:
            logger.info("Load resume...")
            resume_dict = torch.load(cfg.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)

            optim_state_dict = resume_dict["optim_state_dict"]
            optimizer.load_state_dict(optim_state_dict)
    if args.distributed:
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    iter_num = len(train_data_loader) 
    for epoch in range(start_epoch,cfg.runner.max_epochs):
        if epoch + 1 > 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.000001
        epoch_loss = 0
        epoch_loss_C = 0
        model = model.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt0, gt1, meas0, meas1 = data
            gt = gt1.float().to(args.device)
            meas0 = meas0.unsqueeze(1).float().to(args.device)
            meas1 = meas1.unsqueeze(1).float().to(args.device)
            batch_size = meas0.shape[0]

            Phi0 = einops.repeat(mask0,'cr h w->b cr h w',b=batch_size)
            Phi0_s = einops.repeat(mask0_s,'h w->b 1 h w',b=batch_size)

            Phi0 = torch.from_numpy(Phi0).to(args.device)
            Phi0_s = torch.from_numpy(Phi0_s).to(args.device)

            Phi1 = einops.repeat(mask1,'cr h w->b cr h w',b=batch_size)
            Phi1_s = einops.repeat(mask1_s,'h w->b 1 h w',b=batch_size)

            Phi1 = torch.from_numpy(Phi1).to(args.device)
            Phi1_s = torch.from_numpy(Phi1_s).to(args.device)


            optimizer.zero_grad()

            model_out_C, model_out = model((meas0,meas1), (Phi0,Phi1), (Phi0_s,Phi1_s))
            if not isinstance(model_out,list):
                model_out = [model_out]

            loss_C = torch.sqrt(criterion(model_out_C, gt))
            epoch_loss_C += loss_C.item()

            loss = torch.sqrt(criterion(model_out[-1], gt))
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            if rank==0 and (iteration % cfg.log_config.interval) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_len = len(str(iter_num))
                logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}, loss_C: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item(),loss_C.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
            if rank==0 and (iteration % cfg.save_image_config.interval) == 0:
                sing_out = model_out[-1][0].detach().cpu().numpy()
                sing_gt = gt[0].cpu().numpy()
                image_name = osp.join(train_image_save_dir,str(epoch)+"_"+str(iteration)+".png")
                save_image(sing_out,sing_gt,image_name)
        end_time = time.time()
        if rank==0:
            logger.info("epoch: {}, avg_loss: {:.5f}, avg_loss_C: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),epoch_loss_C/(iteration+1),end_time-start_time))

        if rank==0 and (epoch % cfg.checkpoint_config.interval) == 0:
            if args.distributed:
                save_model = model.module
            else:
                save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            torch.save(checkpoint_dict,osp.join(checkpoints_dir,"epoch_"+str(epoch)+".pth")) 

        # if rank==0 and cfg.eval.flag and epoch % cfg.eval.interval==0:
        #     if args.distributed:
        #         psnr_dict,ssim_dict = eval_psnr_ssim(model.module,test_data,(mask0,mask1,mask2),(mask0_s,mask1_s,mask2_s),args)
        #     else:
        #         psnr_dict,ssim_dict = eval_psnr_ssim(model,test_data,(mask0,mask1,mask2),(mask0_s,mask1_s,mask2_s),args)

        #     psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
        #     ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
        #     logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
        #     logger.info("Mean SSIM: \n{}.\n".format(ssim_str))

if __name__ == '__main__':
    args = parse_args()
    # args.work_dir = osp.join('./work_dirs', 'efficientsci')
                             
    cfg = Config.fromfile(args.config)
    cfg.train_data.type = 'DavisMixData'
    cfg.train_data.gene_meas.type = ['GenerationGrayMeas', 'GenerationBayerMeas']
    cfg.gene_meas.type = ['GenerationGrayMeas', 'GenerationBayerMeas']
    cfg.model.color_ch = [1, 3]
    cfg.checkpoints = ['checkpoints/efficientsci_base.pth', 'checkpoints/efficientsci_color_base.pth']

    cfg.train_data.mask_path = './mask/mask.mat'
    cfg.runner.max_epochs = 50
    main(args, cfg)


