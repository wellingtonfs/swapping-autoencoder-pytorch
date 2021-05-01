import torch
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss

class CondPoseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",
                            type=util.str2bool, default=True)
        return parser

    def initialize(self):
        self.M = networks.create_network(self.opt, self.opt.netM, "Mapping")
        self.E = networks.create_network(self.opt, self.opt.netE, "Encoder") #'Encoder' é do condpose, e 'encoder' é o do swapping
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0 and not self.opt.CondPose:
            self.Dpatch = networks.create_network(
                self.opt, self.opt.netPatchD, "patch_discriminator"
            )

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        #pred_amostra = self.D(amostra)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        '''
        losses["D_amostra"] = loss.gan_loss(
            pred_amostra, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        '''

        return losses

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, images, poses):
        self.num_discriminator_iters.add_(1)

        #real: Batch X 3 X 256 X 256
        #pose: Batch X 32 X 256 X 256

        losses, metrics = {}, {}
        B = images.size(0)

        #passando pelo encoder e mapping
        mu, logvar = self.E(images, poses)
        estrutura = self.M(poses)

        #amostrando
        variancia = torch.exp(logvar * 0.5)
        z_latent = self.amostragem_logvar(mu, logvar)

        #reconstrução
        rec = self.G(estrutura, z_latent)

        assert B % 2 == 0, "Batch size must be even on each GPU."

        losses = self.compute_image_discriminator_losses(images, rec)

        if self.opt.lambda_PatchGAN > 0.0 and not self.opt.CondPose:
            patch_losses = self.compute_patch_discriminator_losses(real, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def kl_loss(self, mu, logvar, med, std):
        kl = torch.sum( ((1/2*std.pow(2)) * ((mu-med).pow(2) + logvar.pow(2) - std.pow(2))).abs_() )
        return kl

    def amostragem_logvar(self, mu, variancia):
        if self.opt.num_gpus > 0:
            z_normal = torch.randn(mu.shape[0], mu.shape[1]).to("cuda:0")
        else:
            z_normal = torch.randn(mu.shape[0], mu.shape[1])

        z_latent = z_normal * variancia + enc_mu
        return z_latent

    def compute_generator_losses(self, images, poses):
        #real: Batch X 3 X 256 X 256
        #pose: Batch X 32 X 256 X 256

        losses, metrics = {}, {}
        B = images.size(0)

        #passando pelo encoder e mapping
        mu, logvar = self.E(images, poses)
        estrutura = self.M(poses)

        #amostrando
        variancia = torch.exp(logvar * 0.5)
        z_latent = self.amostragem_logvar(mu, variancia)

        #reconstrução
        rec = self.G(estrutura, z_latent)

        #perdas
        metrics["L1_dist"] = self.l1_loss(rec, real)
        metrics["KL_dirv"] = self.kl_loss(mu, variancia, torch.zeros_like(mu), torch.ones_like(variancia))

        if self.opt.lambda_KL > 0.0:
            losses["KL_dirv"] = metrics["KL_dirv"] * self.opt.lambda_KL

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

        return losses, metrics

    def get_visuals_for_snapshot(self, real):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        sp, gl = self.E(real)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl)
        mix = self.G(sp, self.swap(gl))

        visuals = {"real": real, "layout": layout, "rec": rec, "mix": mix}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False):
        return self.E(image, extract_features=extract_features)

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters()) + list(self.M.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0 and not self.opt.CondPose:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
