import torch


def masks_like(tensor, zero=False, generator=None, p=0.2, current_latent_num=8):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, :-current_latent_num] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, :-current_latent_num]).exp()
                    v[:, :-current_latent_num] = torch.zeros_like(v[:, :-current_latent_num])
                else:
                    u[:, :-current_latent_num] = u[:, :-current_latent_num]
                    v[:, :-current_latent_num] = v[:, :-current_latent_num]
        else:
            for u, v in zip(out1, out2):
                u[:, :-current_latent_num] = torch.zeros_like(u[:, :-current_latent_num])
                v[:, :-current_latent_num] = torch.zeros_like(v[:, :-current_latent_num])

    return out1, out2