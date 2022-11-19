import torch
from torch import nn

class PosEmbedding(nn.Module):
    def __init__(self, k):
        """
        Defines a function that embeds x to (sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.freqs = 2**torch.linspace(0, k-1, k)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)
        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],     # 8 layers with width of 256, and skip at 4
                 in_channels_xyz=63,        # original from nerf
                 in_channels_dir=27,        # original from nerf
                 in_channels_t=48,          # time encoding dimension
                 output_flow=False,         # if outputs fw, bw, disocc
                 flow_scale=0.2):           # how much scale to multiply to flow output (in NDC)
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_t = in_channels_t
        self.output_flow = output_flow
        self.flow_scale = flow_scale

        # static model --------------------------------------------------------
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"static_xyz_encoding_{i+1}", layer)
        self.static_xyz_encoding_final = nn.Linear(W, W)

        # viewdir layer for color
        self.static_dir_encoding = nn.Sequential(
                    nn.Linear(W+in_channels_dir, W), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Linear(W, 1)
        self.static_rgb = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

        # dynamic model -------------------------------------------------------
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+in_channels_t, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz+in_channels_t, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"transient_xyz_encoding_{i+1}", layer)
        self.transient_xyz_encoding_final = nn.Linear(W, W)

        # DIFF from original, we have no dir input here, (it's moving ?

        # dynamic output layers
        self.transient_sigma = nn.Linear(W, 1)
        self.transient_rgb = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

        # predict forward and backward flows
        # DIFF from original, we have no disocclusion here
        if self.output_flow:
            self.transient_flow_fw = nn.Sequential(nn.Linear(W, 3), nn.Tanh())
            self.transient_flow_bw = nn.Sequential(nn.Linear(W, 3), nn.Tanh())


    def forward(self, x, output_static=True, output_transient=True,
                output_transient_flow=[]):
        """
        GO THROUGH THE NETWORK
        Params:
            x: the embedded vector of position (+ direction + transient)
            output_static: whether to use static model and output
            output_transient: whether to use dyn model and output
            output_transient_flow: [] or ['fw'] or ['bw'] or ['fw', 'bw'] or ['fw', 'bw', 'disocc']
        Attention:
            if you are setting output_transient true, the input must have t assumed
            may be optimize this part..
        Outputs (concatenated):
            static: only output_static=True
                static_rgb, static_sigma
            static + dyn: output_static=True output_transient=True
                static_rgb, static_sigma, transient_rgb, transient_sigma
            static + dyn + output_transient_flow:
                above + the chosen item in output_transient_flow
        """
        if output_transient:
            input_xyz, input_dir, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                self.in_channels_t], 1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir], 1)
        
        # static
        if output_static:
            xyz_ = input_xyz
            for i in range(self.D):
                if i in self.skips:
                    xyz_ = torch.cat([input_xyz, xyz_], 1)
                xyz_ = getattr(self, f"static_xyz_encoding_{i+1}")(xyz_)

            # static sigma output
            static_sigma = self.static_sigma(xyz_) # (B,1)

            xyz_ = self.static_xyz_encoding_final(xyz_)
            dir_encoding_input = torch.cat([xyz_, input_dir], 1)
            xyz_ = self.static_dir_encoding(dir_encoding_input)

            # static rgb output
            static_rgb = self.static_rgb(xyz_) # (B, 3)

            static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

            if not output_transient:
                return static # (B, 4)

        # dyn
        xyz_ = torch.cat([input_xyz, input_t], 1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, input_t, xyz_], 1)
            xyz_ = getattr(self, f"transient_xyz_encoding_{i+1}")(xyz_)
        xyz_ = self.transient_xyz_encoding_final(xyz_)

        # dyn sigma output
        transient_sigma = self.transient_sigma(xyz_)
        # dyn rgb output
        transient_rgb = self.transient_rgb(xyz_)

        transient_list = [transient_rgb, transient_sigma] # (B, 4)

        if 'fw' in output_transient_flow:
            transient_list += [self.flow_scale * self.transient_flow_fw(xyz_)]
        if 'bw' in output_transient_flow:
            transient_list += [self.flow_scale * self.transient_flow_bw(xyz_)]

        transient = torch.cat(transient_list, 1) # (B, 10)
        if output_static:
            return torch.cat([static, transient], 1) # (B, 14)
        return transient # (B, 10)
