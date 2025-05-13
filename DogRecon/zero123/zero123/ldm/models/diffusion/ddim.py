"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
from PIL import Image 
import cv2
from sklearn.cluster import KMeans
from matplotlib import cm
import matplotlib.pyplot as plt

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.models.diffusion.sampling_util import renorm_thresholding, norm_thresholding, spatial_norm_thresholding


"""# TODO: thresholding 기준
def compute_l2_distances(image):
    # 이미지 크기
    h, w, c = image.shape
    l2_distances = np.zeros((h, w), dtype=np.float32)

    # 각 픽셀에 대해 인근 픽셀과의 L2 거리 계산
    for i in range(1, h-1):
        for j in range(1, w-1):
            # 현재 픽셀의 벡터
            current_pixel = image[i, j]

            # 인근 픽셀의 벡터들
            neighbors = [
                image[i-1, j], image[i+1, j],
                image[i, j-1], image[i, j+1],
                image[i-1, j-1], image[i-1, j+1],
                image[i+1, j-1], image[i+1, j+1]
            ]

            # L2 거리 계산
            distances = [np.linalg.norm(current_pixel - neighbor) for neighbor in neighbors]
            l2_distances[i, j] = np.mean(distances)

    return l2_distances"""

class DDIMSampler(object):
    def __init__(self, model, schedule="linear",initial_weight=0.8, kernel_size=15, **kwargs):
    #def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()    
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.initial_weight = initial_weight
        self.kernel_size = kernel_size

    def to(self, device):
        """Same as to in torch module
        Don't really underestand why this isn't a module in the first place"""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_v = getattr(self, k).to(device)
                setattr(self, k, new_v)


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    #결국여기확인
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None, mask_36=None,white_mask=None,image_name=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,mask_36=mask_36, white_mask=white_mask,image_name=image_name,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      t_start=-1, mask_36=None, white_mask=None, image_name=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        timesteps = timesteps[:t_start]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        #white_list = []

        #mask_36
        raw_image =Image.fromarray(mask_36)
        resize_image = raw_image.resize((32,32),Image.BILINEAR)
        resize_np = np.array(resize_image)
        l1_mask = torch.from_numpy(resize_np).to(device=device, dtype=img.dtype)
        
        # 팽창한 마스크 생성
        """kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        new_raw_image = Image.fromarray(mask_36)
        new_resize_image = new_raw_image.resize((32, 32), Image.BILINEAR)
        new_resize_np = np.array(new_resize_image)
        big_mask = torch.from_numpy(new_resize_np).to(device=device, dtype=img.dtype)"""
        #######################
        # 팽창된 마스크
        white_area = mask_36 == 1

        #kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        print(f'kernel_size : {kernel.shape}')

        # expand 시작
        big_mask = cv2.dilate(white_area.astype(np.uint8), kernel, iterations=1)

        # 원래 mask 크기로 유지
        big_mask_np = big_mask * 1  # 흰색 부분 유지
        #######################

        dilated_l1_mask = torch.from_numpy(big_mask_np).to(device=device, dtype=img.dtype)
        

        # out_mask 및 background_mask 생성
        out_mask = big_mask_np - mask_36
        
        raw_image_out_mask =  Image.fromarray(out_mask)
        resize_image_out_mask  = raw_image_out_mask.resize((32,32),Image.BILINEAR)
        resize_out_mask = np.array(resize_image_out_mask)
        dilated_out_mask = torch.from_numpy(resize_out_mask).to(device=device, dtype=img.dtype)
        
        background_mask = 1 - big_mask_np
        
        raw_image_background =Image.fromarray(background_mask)
        resize_image_background = raw_image_background.resize((32,32),Image.BILINEAR)
        resize_background = np.array(resize_image_background)
        dilated_background_mask = torch.from_numpy(resize_background).to(device=device, dtype=img.dtype)
        
    
        
        #l2_mask
        """raw_image2 =Image.fromarray(lv2_mask)
        resize_image2 = raw_image2.resize((32,32),Image.BILINEAR)
        resize_np2 = np.array(resize_image2)
        l2_mask = torch.from_numpy(resize_np2).to(device=device, dtype=img.dtype)
        
        #l3_mask
        raw_image3 =Image.fromarray(lv3_mask)
        resize_image3 = raw_image3.resize((32,32),Image.BILINEAR)
        resize_np3 = np.array(resize_image3)
        l3_mask = torch.from_numpy(resize_np3).to(device=device, dtype=img.dtype)
        
        #back_mask
        raw_image4 =Image.fromarray(inv_mask)
        resize_image4 = raw_image4.resize((32,32),Image.BILINEAR)
        resize_np4 = np.array(resize_image4)
        back_mask = torch.from_numpy(resize_np4).to(device=device, dtype=img.dtype)"""
        
        #test_mask = l2_mask + l3_mask

        white_mas = torch.from_numpy(white_mask).to(device=device, dtype=img.dtype)
        
        
        #디버깅
        #print(f'test_mask:{test_mask}')
        #print(f'out_mask.shape:{out_mask}')
        #print(f'dilated_out_mask.shape:{dilated_out_mask}')
        print(f'img.shape: {img.shape}')
        print(f'dilated_out_mask.shape: {dilated_out_mask.shape}')
        #print(f'test_mask.shape: {test_mask.shape}')


        #모든 mask가 1인지 확인한다. -> (mask없을 때 처리)
        #all_ones = torch.all(l1_mask == 1) and torch.all(l2_mask == 1) and torch.all(l3_mask == 1) and torch.all(back_mask == 1)
        
        #hyperparameter(l2_weight. l3_weight)
        #l2_weight = 0.7
        #l3_weight = 0.3
        new_white_area = big_mask == 1
        eroding_kernel = self.kernel_size
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            #original
            #if i == 25:
            #    img = img*resize_torch + (0.5*white_mas[i]+0.5*img)*(1-resize_torch) 
            
            #if i == 25:
            #    img = img*resize_torch
            #if i == 27:
                #img = img*resize_torch + 0.7*white_mas[i]*(1-resize_torch)
                #img = img*resize_torch + (0.999*white_mas[i])*(1-resize_torch)
            
            if 0<=i<=30:    
                """L1_mask = img * l1_mask 
                L2_mask = ((l2_weight) * (img * l2_mask)) + ((1 - l2_weight) * (white_mas[i] * l2_mask))
                L3_mask = ((l3_weight) * (img * l3_mask)) + ((1 - l3_weight) * (white_mas[i] * l3_mask))
                Background = white_mas[i] * back_mask
                
                img = L1_mask + L2_mask + L3_mask + Background"""
            #if 1 <= i <= 30:
                # i 값에 따라 흰색 부분의 boundary를 줄이자
                #new_big_mask = torch.from_numpy(cv2.erode(dilated_l1_mask_np, kernel, iterations=i//3)).to(device=device, dtype=img.dtype)
                #iter_out_mask = new_big_mask - l1_mask
                #iter_background_mask = 1 - new_big_mask
                
                # weight 값 줄이기
                #weight = self.initial_weight * (1 - (i / 30))
                
                # Erode the out_mask and background_mask gradually
                #eroding_kernel = self.kernel_size - (5 * (i / 31)) # 아.. cv.dilate에서 kernel은 정수밖에 안됨 ㅠ
                #eroding_kernel = self.kernel_size
                if i % 5 == 0 and i != 0:  # i가 5의 배수일 때마다
                    eroding_kernel -= 1 
                erode_kernel = np.ones((eroding_kernel, eroding_kernel), np.uint8)
                print(f'erode_kernel : {erode_kernel.shape}')

                eroded_out_mask = cv2.dilate(white_area.astype(np.uint8), erode_kernel, iterations=1)

                # 원래 mask 크기로 유지
                eroded_out_mask_np = eroded_out_mask * 1  # 흰색 부분 유지
                #######################

                eroded_l1_mask = torch.from_numpy(eroded_out_mask_np).to(device=device, dtype=img.dtype)
                

                # out_mask 및 background_mask 생성
                eroded_out_mask = eroded_out_mask_np - mask_36
                
                e_raw_image_out_mask =  Image.fromarray(eroded_out_mask)
                e_resize_image_out_mask  = e_raw_image_out_mask.resize((32,32),Image.BILINEAR)
                e_resize_out_mask = np.array(e_resize_image_out_mask)
                eroding_out_mask = torch.from_numpy(e_resize_out_mask).to(device=device, dtype=img.dtype)
                
                eroded_back_mask = 1 - eroded_out_mask_np
                
                e_raw_image_background =Image.fromarray(eroded_back_mask)
                e_resize_image_background = e_raw_image_background.resize((32,32),Image.BILINEAR)
                e_resize_background = np.array(e_resize_image_background)
                eroded_background_mask = torch.from_numpy(e_resize_background).to(device=device, dtype=img.dtype)
                
                ###############
                                
                
                weight = self.initial_weight - (0.5 * (i / 31))
                
                print(f'weight:{weight}')

                L1_mask = img * l1_mask
                L2_mask = ((weight) * (img * eroding_out_mask)) + ((1 - weight) * (white_mas[i] * eroding_out_mask)) #원래는 dilated_out_mask
                Background = white_mas[i] * eroded_background_mask
                
                img = L1_mask + L2_mask + Background
                
                #iter_out_mask_np = iter_out_mask.cpu().numpy() * 255
                #iter_out_mask_img = Image.fromarray(iter_out_mask_np.astype(np.uint8))
                
                # iter_out_mask_img.save(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/dh_images_bm/{i:04d}_big_mask.png')

    
                
                """L1_mask = img * l1_mask
                L2_mask = ((l2_weight) * (img * l2_mask)) + ((1 - l2_weight) * (white_mas[i] * l2_mask))
                L3_mask = ((l3_weight) * (img * l3_mask)) + ((1 - l3_weight) * (white_mas[i] * l3_mask))
                Background = white_mas[i] * back_mask
                
                img = L1_mask + L2_mask + L3_mask + Background"""
        
            #if i == 36:
            #    img = img*resize_torch + white_mas[i]*(1-resize_torch)



            #if i <20:
            #    img = img*resize_torch + white_mas[i]*(1-resize_torch)
            #
            #print(resize_torch)
           #if i == 25:
            #if i == 20:
            #    img = img*resize_torch #+ 0.01*white_mas[i]*(1-resize_torch)
                #img = img*resize_torch + (0.5*white_mas[i]+0.5*img)*(1-resize_torch) 
            #if i in [48]:
                #img = img*resize_torch + (0.5*white_mas[i]+0.5*img)*(1-resize_torch)  #너무 세게 주지 말자.
                #img = img*resize_torch + (0.9*white_mas[i]+0.1*img)*(1-resize_torch)
            #    img = img*resize_torch + white_mas[i]*(1-resize_torch)
            #35,37
            #46 너무 흐림 괜찮긴한데.
            #43, 46보다는 많이 괜찮은데 아직 흐림.
            #40, 46 보다 흐린건 없어졌ㅈ는데 형태가 생기면서 이상해진다.
            #38 지금까지 가장 좋은결과 아 진짜 애매하다.
            #37.. 진짜 애매... 왼쪽에 조금만.. .하하.
            #36은 .. 아 진짜.. 다시 원상복귀다.\
            #26,그대로
            #40하니까 다 사라진다.
            #35도 뭐비슷하긴하다. 이거 없애면 그냥 흐린 이미지만 나오니까..
            #35나 30이나 ㄷ;; 27이나 비슷하다.. 뒤에 마스크를 조금 더 위로 올려보자.
            #애를 바꿔보자.
            #27에서 다리가 없어졌다가 바로28 되니까 생김.
            #48이나 46이나 큰차이. x 오.. 40 나쁘지 않다. 39 더나오긴한데 35는 완전 그대로다.. 와.. 37 딱 좋긴한데 다리가조금더 길었으면. 
            # 36에서 37로 바뀌니까 다리가 더 짧아지긴함.
            # 35로 바꾸니까 그냥 다리가 바로 생김. 그냥 그대로네.
            
            
            
            #if i == 34:
            #    img = img*resize_torch + white_mas[i]*(1-resize_torch)
            #오 46 good.
            #if i == 36:
            #    img = img*resize_torch + white_mas[i]*(1-resize_torch)
            #img = img*resize_torch + white_mas[i]*(1-resize_torch)
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,mask_36=mask_36, image_name=image_name)
            img, pred_x0 = outs
            #white_list.append(img.cpu().numpy())
            if callback:
                img = callback(i, img, pred_x0)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        #np.save('./white_dog.npy',np.array(white_list))
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, mask_36=None,image_name=None):
        b, *_, device = *x.shape, x.device

        #print(f'index={index}, t={t}')
        

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            #i  <5 
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2) 
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        #print(f'x_prev.shape={x_prev.shape}') (1,4,32,32)
        #if index ==25:
        #    x_prev = x_prev*resize_torch
        #if index == 35:
        #    x_prev = x_prev*resize_torch
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec