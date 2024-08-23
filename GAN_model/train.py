import torch
from embeddings import get_caption_embedding, get_clip_model, get_similarity_score
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch.nn as nn

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def tensor_to_image(tensor):
    # Convert from range [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert from [0, 1] to [0, 255] and cast to uint8
    tensor = tensor * 255
    tensor = tensor.clamp(0, 255).byte()
    # Convert tensor to PIL image
    return tensor
    # if tensor.shape[0] == 1:  # If grayscale
    #     tensor = tensor.squeeze(0)
    #     return transforms.ToPILImage()(tensor)
    # else:  # If RGB
    #     return transforms.ToPILImage()(tensor)
    
from PIL import Image

from PIL import Image

def mode_seeking_loss(fake_image1, fake_image2, z1, z2, lambda_diversity=1.0):
    # Use L2 norm instead of L1 to avoid the use of `sgn`
    image_diff = torch.mean((fake_image1 - fake_image2) ** 2)
    z_diff = torch.mean((z1 - z2) ** 2)
    loss = lambda_diversity * image_diff / z_diff
    return loss

def save_image(tensor, path):
    # Convert tensor to numpy array and normalize to [0, 255]
    image = tensor.permute(1, 2, 0).numpy()  # Convert to HWC
    image = (image * 255).astype('uint8')  # Scale to [0, 255]
    Image.fromarray(image).save(path)

def generate_and_save_images(epoch, test_input, caption_embeddings, save_dir="GAN_model/generated_images", generator=None):
    generated_images = generator(test_input, caption_embeddings).detach().cpu()
    
    generated_images = (generated_images + 1) / 2  # Normalize to [0, 1]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, img in enumerate(generated_images):
        save_image(img, f"{save_dir}/epoch_{epoch}_image_{i}.png")

def compute_gradient_penalty(D, real_samples, fake_samples, caption_embeddings):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, caption_embeddings)
    fake = torch.ones(d_interpolates.shape).to(real_samples.device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(num_epochs, dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, batch_max = 50, preSave = False):
    # Check if MPS (Metal Performance Shaders) is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple GPU) is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU. MPS not available.")
    
    generator.to(device)
    discriminator.to(device)

    label_to_class = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
    #apply weight initializations
    generator.apply(weights_init_kaiming)
    discriminator.apply(weights_init_xavier)

    if(preSave and os.path.exists('GAN_model/weights/generator.pth') and os.path.exists('GAN_model/weights/discriminator.pth')):
        generator.load_state_dict(torch.load('GAN_model/weights/generator.pth'))
        discriminator.load_state_dict(torch.load('GAN_model/weights/discriminator.pth'))
        print("Generator and discriminator weights loaded.")

    clip_model, clip_processor = get_clip_model()

    max_norm = 1.0

    clip_model.to(device)
    captions_names = []
    for epoch in range(num_epochs):
        print ("starting epoch " + str(epoch))
        i = 0
        for real_images, captions in dataloader:
            if i >= batch_max:
                print("breaking at batch_max", batch_max)
                break
            i = i+1
            #get the embeddings for the captions
            real_images = real_images.to(device)
            
            # #testing show one image and caption pair
            # print(captions[0])
            # plt.imshow(real_images[0].permute(1, 2, 0).cpu())
            # plt.show()
            
            captions_names = [label_to_class[label.item()] for label in captions]

            caption_embeddings = get_caption_embedding(captions_names, clip_model, clip_processor).to(device)

            batch_size = real_images.size(0)

            #chanigng this 
            real_labels = (torch.ones(batch_size) * 0.9).unsqueeze(1).to(device) # Instead of 1.0
            fake_labels = (torch.zeros(batch_size) + 0.1).unsqueeze(1).to(device)  # Instead of 0.0

            #train the discriminator
            optimizer_d.zero_grad()

            #train the discriminator on real images, inputting the captions along with the real images
            output = discriminator(real_images, caption_embeddings).to(device)
            loss_real = criterion(output, real_labels).to(device)

            #generate the fake images with Generator
            # print(torch.randn(batch_size, 100))
            # print(caption_embeddings.shape)
            outputIM = generator(torch.randn(batch_size, 256).to(device), caption_embeddings).to(device) # Detach to avoid training the generator here
            output = discriminator(outputIM.detach(), caption_embeddings).to(device)
            
            loss_fake = criterion(output, fake_labels).to(device)
            total_loss = (loss_real + loss_fake).to(device)

            # Assume lambda_gp is a hyperparameter for gradient penalty
            lambda_gp = 10
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, outputIM.detach(), caption_embeddings)
            total_loss += lambda_gp * gradient_penalty
            
            
            #if(i%2 == 0):
            total_loss.backward()

            #gradient clipping
            #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)


            optimizer_d.step()

            #train the generator
            optimizer_g.zero_grad()
            output = discriminator(outputIM, caption_embeddings).to(device)

            #additional loss with similarity score
            #get similarity score
            # Compute similarity loss
            similarity_loss = torch.tensor(0.0)
            
            for j in range(batch_size):
                image = outputIM[j].detach()
                #check if image is normalized
                image = tensor_to_image(image).to(device)
                #image = image.to(device)
                caption = captions_names[j]
                similarity = get_similarity_score(image, caption, clip_model, clip_processor)
                #print(similarity)
                similarity_loss += (1 - similarity)  # Minimize the distance
            
            similarity_loss /= batch_size

            similarity_loss = similarity_loss.to(device)

            #diversity loss
            # z1, z2 = torch.randn(batch_size, 256).to(device), torch.randn(batch_size, 256).to(device)
            # fake_image1, fake_image2 = generator(z1, caption_embeddings), generator(z2, caption_embeddings)
            
            # diversity_loss = mode_seeking_loss(fake_image1, fake_image2, z1, z2).to(device)
            
            # print(f"Output shape: {output.shape}")
            # print(f"Real labels shape: {real_labels.shape}")

            # Forward pass through discriminator to get features for real and fake images
            fake_images = generator(torch.randn(batch_size, 256).to(device), caption_embeddings)
            real_features = discriminator.extract_features(real_images, caption_embeddings)
            fake_features = discriminator.extract_features(fake_images, caption_embeddings)

            # Compute feature matching loss (MSE between real and fake features)
            fm_loss = torch.mean((real_features - fake_features) ** 2).to(device)
            
            g_loss = (criterion(output, real_labels) + 1 * fm_loss + 0.2 * similarity_loss).to(device) #can change
            
            g_loss.backward()
            #print(f"Generator loss: {g_loss.item()}")

            #gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm)

            optimizer_g.step()
            
            if(i %5 == 0):
                print(f"Epoch [{epoch}/{num_epochs}], Loss D: {total_loss.item()}, Loss G: {g_loss.item()}")
            
            i += 1


        print("finished epoch" + str(epoch))
        # Save the model checkpoint after 10 epochs
        if (epoch+1) % 5 == 0:
            torch.save(generator.state_dict(), 'GAN_model/weights/generator.pth')
            torch.save(discriminator.state_dict(), 'GAN_model/weights/discriminator.pth')
            print("Model checkpoints saved.")

        if epoch % 1 == 0:
            # Generate and save images every 5 epochs
            #test_caption = "A futuristic cityspace at night with flying cars"
            print("Captions:", captions_names[0:20])
            input_captionEmbedding = get_caption_embedding(captions_names[0:20], clip_model, clip_processor).to(device)
            generate_and_save_images(epoch, torch.randn(20, 256).to(device), input_captionEmbedding, generator=generator)