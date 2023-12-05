from diffusers import StableDiffusionXLPipeline

if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    )

    image_size = 64
    image = pipe.fine_tune(image_size=image_size, num_inference_steps=40).images[0]
