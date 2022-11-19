import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--root_dir', type=str, required=True, help='root directory of dataset')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 288], help='resolution (img_w, img_h) of the image')
    parser.add_argument('--start_end', nargs='+', type=int, default=[0, 30], help='start and end frames, you may set as [0, #frames]')

    # Original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10, help='number of features in xyz embedding')
    parser.add_argument('--N_emb_dir', type=int, default=4, help='number of features in dir embedding')
    parser.add_argument('--N_samples', type=int, default=128, help='number of coarse samples')
    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0, help='std dev of noise added to regularize sigma')
    # Additional model parameters
    parser.add_argument('--N_tau', type=int, default=48, help='number of embeddings for transient/dyn objects')
    parser.add_argument('--lambda_geo_init', type=float, default=0.04, help='optical flow consistency loss coefficient')
    parser.add_argument('--thickness', type=int, default=1, help='prior about dynamic object thickness (how many intervals objects occupy)')
    parser.add_argument('--flow_scale', type=float, default=0.2, help='flow scale to multiply to flow network output')

    # Size
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024, help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam',help='optimizer type',choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',help='scheduler type',choices=['const', 'steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,help='learning rate decay amount')
    ###########################
    #### params for poly ######
    parser.add_argument('--poly_exp', type=float, default=0.9,help='exponent for polynomial learning rate decay')
    ###########################

    return parser.parse_args()
