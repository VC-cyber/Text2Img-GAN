from diffusers import StableDiffusionPipeline

# Load the pipeline (model)
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

#pipe = pipe.to("cuda")

# Input your prompt
prompt = "dad making omlette in the kitchen"

print("Generating image...")
# Generate the image
image = pipe(prompt).images[0]

print("Saving Image...")
# Save or display the image
image.save("GeneratedImages/generated_image3.png")

print("Image saved successfully!")