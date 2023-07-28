import torch

class DiffusionModel():
    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps = 1000):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps

        self.betas = cosine_beta_schedule(timesteps=timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
    
    def forward(self, x_0, t):
        ## x_0: (batch_size, num_images, height, width), t: (batch_size)
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_t_index(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_t_index(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
            
        mean = sqrt_alphas_cumprod_t * x_0
        variance = sqrt_one_minus_alphas_cumprod_t* noise
        
        return mean + variance, noise

    
    def backward(self, x, t, model, **kwargs):

        betas_t = self.get_t_index(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_t_index(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_t_index(torch.sqrt(1.0 / self.alphas), t, x.shape)

        ##EQ 11:
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = betas_t

        if t == 0:
            variance=0
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise 
        
        return mean + variance
    
    def get_t_index(values, t, x_shape):
        batch_size = t.shape[0]
        result = values.gather(-1, t.cpu())
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    

    def cosine_beta_schedule(timesteps, s=0.008):
        ###cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    
