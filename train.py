# general
from opt import get_opts
import torch
import copy
from collections import defaultdict

# dataset
from torch.utils.data import DataLoader
from datasets.monocular import MonocularDataset

# models
from models.nerf import PosEmbedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *
from torchvision.utils import make_grid

# losses and metrics
from losses import NeRFWLoss
import metrics

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)


class NSFFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # models details ------------------------------------------------------------
        # original embedding of pos and dir as in the nsff/nerf and add t embedding
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir)
        self.N_frames = self.hparams.start_end[1]-self.hparams.start_end[0]
        self.embedding_t = torch.nn.Embedding(self.N_frames, hparams.N_tau)

        self.embeddings = {'xyz': self.embedding_xyz,   # a class
                           'dir': self.embedding_dir,   # a class
                           't':   self.embedding_t}     # a nnEmbedding
        self.output_transient_flow = ['fw', 'bw', 'disocc']     # we output them all

        # have only one model
        self.nerf_fine = NeRF(typ='fine',
                              in_channels_xyz=6*hparams.N_emb_xyz+3,
                              in_channels_dir=6*hparams.N_emb_dir+3,
                              in_channels_t=hparams.N_tau,
                              output_flow=True,
                              flow_scale=hparams.flow_scale)
        self.models = {'fine': self.nerf_fine}
        self.models_to_train = [self.models, self.embedding_t]  # only train t

        # losses and metrics --------------------------------------------------------
        self.loss = NeRFWLoss(lambda_geo=self.hparams.lambda_geo_init,
                              thickness=self.hparams.thickness)

    # hide val number, it's meaningless
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, test_time=False, **kwargs):
        """
            Do batched inference on rays using chunk.
        Input:
            rays:   (B,6)
            ts:     (B)
            test_time:  True if val, False if train
            **kwargs:   info bring to render
        """
        B = rays.shape[0]
        results = defaultdict(list)
        kwargs_ = copy.deepcopy(kwargs)
        for i in range(0, B, self.hparams.chunk):
            # in validation, B > chunk
            # in training, default B = 512 < default chunk 32768
            rendered_ray_chunks = \
                render_rays(self.models,                    
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.train_dataset.N_frames-1,
                            self.hparams.N_samples,
                            self.hparams.perturb if not test_time else 0,   # dont perturb
                            self.hparams.noise_std if not test_time else 0, # for val
                            self.hparams.chunk//4 if test_time else self.hparams.chunk, # and less chunk
                            **kwargs_)

            for k, v in rendered_ray_chunks.items():
                if test_time: v = v.cpu()
                results[k] += [v]   # now results take all the dict key
        for k, v in results.items(): results[k] = torch.cat(v, 0)   # chunk to batch
        return results  # results have all render_rays output keys with result (B/C,its_size)

    def setup(self, stage):
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  'start_end': tuple(self.hparams.start_end)}
        self.train_dataset = MonocularDataset(split='train', **kwargs)
        self.val_dataset = MonocularDataset(split='val', **kwargs)

        # buffer for flow, check the losses.py
        self.loss.register_buffer('Ks', self.train_dataset.Ks)
        self.loss.register_buffer('Ps', self.train_dataset.Ps)
        self.loss.max_t = self.N_frames-1

    def configure_optimizers(self):
        kwargs = {}
        self.optimizer = get_optimizer(self.hparams, self.models_to_train, **kwargs)
        if self.hparams.lr_scheduler == 'const': return self.optimizer

        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        self.train_dataset.batch_size = self.hparams.batch_size # get_item select rays
        # disable automatic batching as batch_size= None
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        # disable automatic batching as batch_size= None
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def on_train_epoch_start(self):
        self.loss.lambda_geo_d = self.hparams.lambda_geo_init * 0.1**(self.current_epoch//10)
        self.loss.lambda_geo_f = self.hparams.lambda_geo_init * 0.1**(self.current_epoch//10)

    def training_step(self, batch, batch_nb):
        # batch is a dict with items (B,..) # default B 512
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        # rays: (B,6) rgbs: (B,3) ts: (B)
        kwargs = {'epoch': self.current_epoch,  # lightning module counting epoch for LOSS !!
                  'output_transient': True,
                  'output_transient_flow': self.output_transient_flow}
        results = self(rays, ts, **kwargs)  # check rendering short summary for what it has

        loss_d = self.loss(results, batch, **kwargs) # dict with 11 losses
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = metrics.psnr(results['rgb_fine'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items(): self.log(f'train/{k}', v)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        batch['rgbs'] = rgbs = rgbs.cpu() # (H*W, 3)
        if 'mask' in batch: mask = batch['mask'].cpu() # (H*W)
        if 'disp' in batch: disp = batch['disp'].cpu() # (H*W)
        kwargs = {'output_transient': True,
                  'output_transient_flow': []}
        results = self(rays, ts, test_time=True, **kwargs)

        # compute error metrics
        W, H = self.hparams.img_wh
        img = torch.clip(results['rgb_fine'].view(H, W, 3).cpu(), 0, 1)
        img_ = img.permute(2, 0, 1)
        img_gt = rgbs.view(H, W, 3).cpu()

        rmse_map = ((img_gt-img)**2).mean(-1)**0.5
        rmse_map_blend = blend_images(img_, visualize_depth(-rmse_map), 0.5)

        ssim_map = metrics.ssim(img_gt, img, reduction='none').mean(-1)
        ssim_map_blend = blend_images(img_, visualize_depth(-ssim_map), 0.5)

        depth = visualize_depth(results['depth_fine'].view(H, W))
        img_list = [img_gt.permute(2, 0, 1), img_, depth]

        # output transient
        img_list += [visualize_mask(results['transient_alpha_fine'].view(H, W))]
        img_list += [torch.clip(results['_static_rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu(), 0, 1)]
        img_list += [visualize_depth(results['_static_depth_fine'].view(H, W))]

        if 'mask' in batch: img_list += [visualize_mask(1-mask.view(H, W))]
        if 'disp' in batch: img_list += [visualize_depth(-disp.view(H, W))]
        img_grid = make_grid(img_list, nrow=3) # 3 images per row
        self.logger.experiment.add_image('reconstruction/decomposition', img_grid, self.global_step)
        self.logger.experiment.add_image('error_map/rmse', rmse_map_blend, self.global_step)
        self.logger.experiment.add_image('error_map/ssim', ssim_map_blend, self.global_step)

        log = {'val_psnr': metrics.psnr(results['rgb_fine'], rgbs),
               'val_ssim': ssim_map.mean()}
        if (mask==0).any():
            log['val_psnr_mask'] = metrics.psnr(results['rgb_fine'], rgbs, mask==0)
            log['val_ssim_mask'] = ssim_map[mask.view(H, W)==0].mean()

        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim)

        if all(['val_psnr_mask' in x for x in outputs]):
            mean_psnr_mask = torch.stack([x['val_psnr_mask'] for x in outputs]).mean()
            mean_ssim_mask = torch.stack([x['val_ssim_mask'] for x in outputs]).mean()
            self.log('val/psnr_mask', mean_psnr_mask, prog_bar=True)
            self.log('val/ssim_mask', mean_ssim_mask)

def main(hparams):
    system = NSFFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}', filename='{epoch:d}',
                              save_top_k=-1)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[ckpt_cb],
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=1,
                      num_nodes=1,
                      num_sanity_val_steps=1,
                      reload_dataloaders_every_epoch=True,
                      benchmark=True,
                      profiler="simple")

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)