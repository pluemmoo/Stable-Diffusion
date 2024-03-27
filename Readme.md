# Stable Diffusion
## Latent Difussion Model
<p align="center">
  <img src="assets\Latent Diffusion Model.png">
</p>


## Variational Autoencoder

### Encoder

The encoder is a neural network responsible for taking the input data and compressing it into a lower-dimensional latent representation. This latent space captures the essential features of the data while discarding noise and redundancy.

- In most encoders, there's trend of increasing the number of features (channels) while decreasing the image size.
  
#### Typical encoder behavior

- Early layers: The initial layers in the encoder typically increase the number of feature channels. This allows them to capture a wider range of features from the input image.
- Downsampling: Alongside increasing features, the encoder often reduces the image size through techniques like strided convolutions or pooling operations. This reduces the spatial resolution but allows the network to focus on capturing higher-level features.
- After the number of channels increases progressively through several convolutional and residual blocks. In the last convolution layer (Bottleneck layer) there's a deliberate decrease in features eg. from 512 to 8 features (channels). This reduction serves the purpose of compressing the extracted features into a lower-dimensional latent representation.

### Does it means that the features that we increase is totally loss when reduced the features?

- Latent Representation: The bottleneck layer           compresses the extracted features into a lower-dimensional latent space. This compressed representation doesn't explicitly store all the individual features from earlier layers.
- Learning a Compact Representation: The encoder, through its training process, learns to encode the most important information from the increased features into the limited number of channels in the bottleneck layer. This process involves discarding redundancy and focusing on capturing the essence of the image.
- Implicit Information: While the individual features might not be directly preserved, the information they contained is implicitly embedded in the latent representation. The encoder essentially learns a more efficient way to represent the image using a smaller set of features.

### However, there are some limitations

- Reducing features too drastically might lead to information loss. The chosen number of channels in the bottleneck needs to be sufficient to capture the essential information from the previous layers.
- The compressed representation might not perfectly reconstruct the original image due to the discarded details.

### Forward pass in the encoder

    ```py
    for module in self:

        if getattr(module, 'stride', None) == (2, 2):  
        # Padding at downsampling should be asymmetric (see #8)
        # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
        # Pad with zeros on the right and bottom.
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
        x = F.pad(x, (0, 1, 0, 1))

        x = module(x)
    ```

This part of the code deals with applying asymmetric padding before downsampling layers in a convolutional neural network (CNN) architecture

- Looping Through Modules:
  - The for module in self: loop iterates through each module within the Encoder class, which likely consists of convolutional layers and residual blocks.

- Checking for Downsampling:
  - Inside the loop, the code checks if the current module has a stride attribute. The stride attribute in a convolutional layer defines how much the input is shifted by horizontally and vertically after applying the convolution operation.
    - The getattr(module, 'stride', None) part retrieves the stride attribute of the current module. If the attribute doesn't exist, it returns None.
    - The condition if getattr(module, 'stride', None) == (2, 2): specifically checks if the stride is a tuple equal to (2, 2). This indicates a downsampling layer that reduces the spatial dimensions of the input by half in both height and width.

- Asymmetric Padding:

  - If the condition is met, signifying a downsampling layer, the code applies asymmetric padding using x = F.pad(x, (0, 1, 0, 1)). Here's what this means:
    - F.pad is a function from the torch.nn.functional module used for padding tensors with a constant value.
    - x is the input tensor that needs padding before being passed to the downsampling layer.
    - The padding configuration (0, 1, 0, 1) defines the amount of padding to add on each side of the input. In this case: (Have Some Misunderstanding Here, waiting for fixing)
      - 0 for the left and right padding means no padding is added horizontally.
      - 1 for the top and bottom padding adds one element of padding on both the top and bottom of the input.

### The rationale behind asymmetric padding

- When using a convolution with stride 2, the output has half the height and width of the input.
- Symmetric padding (adding equal padding on all sides) can lead to information loss at the borders of the input, especially for small feature maps.
- Asymmetric padding by adding only one element on the top and bottom ensures that all information from the input is preserved, even after downsampling. This can be crucial for capturing important features in the data.

- Applying the Module:
  - After applying asymmetric padding if necessary, the code then passes the padded input x to the current module x = module(x). This could be a convolutional layer or a residual block that performs the actual feature extraction or transformation on the input data.

In essence, the padding serves as a "<b>safety net</b>" to prevent information loss during downsampling, not as a way to increase features. This helps maintain a more accurate representation of the image even when its size is reduced.

In summary, we applied the padding when we do downsampling. Downsampling reduces the spatial dimensions (height and width) of the image while extracting features.

    ```py
    mean, log_variance = torch.chunk(x, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)   
    variance = log_variance.exp()
    stdev = variance.sqrt()
    x = mean + stdev * noise

    x *= 0.18215
    return x
    ```

This part of the code deals with generating a latent representation in a Variational Autoencoder.

- Splitting the Signal
  - Imagine you have a signal x that contains information about the image you're processing.
  - The code uses torch.chunk to split this signal into two parts: mean and log_variance. Think of them as separate channels carrying different information.

- Clamping the Variance
  - The log_variance part represents the "spread" of the information in the signal. Taking the logarithm helps to compress the range of values, making them easier to handle for the network and preventing these stability issues. However, extremely high or low values can cause problems during training.
  - The code uses torch.clamp to act like a safety net. It limits the values in log_variance to be between ==-30 and 20==, ensuring they stay within a reasonable range.

- From Variance to Standard Deviation:
  - The variance is calculated by taking the exponent of the clamped log_variance. This is like raising log_variance to a certain power to get the actual "spread."
  - Then, the code calculates the stdev (standard deviation) by taking the square root of the variance. Standard deviation tells you how much the information in the signal tends to deviate from the average

- Adding Noise for Creativity:
  - VAEs are used to compress and recreate images. This part injects some "creativity" into the process.
  - noise represents a random signal, like static on a TV.
  - The code adds the product of stdev (spread) and noise (randomness) to the mean (average) information. This injects some random variation based on the "spread" of the original signal.

The formula x = mean + stdev * noise is a direct consequence of the reparameterization trick in VAEs. It allows for controlled noise injection based on the data's variability, leading to a more informative latent representation for reconstruction and learning. However, in the normal disturbution, it need to be variance instead of standard derivation. This is a slight technical inaccuracy, but it's a common practice to use "standard deviation" even when referring to the square root of the calculated variance value.

- Scaling the Output:
  - Finally, the code multiplies the entire signal (x) by a constant factor (0.18215). This can be related to normalization for better training behavior.

- Returning the Signal:
  - After all these transformations, the code returns the modified signal x, which now contains a combination of the original information, some randomness based on its spread, and a scaling factor. This signal becomes a part of the VAE's compressed representation of the image.

<BR>
<BR>
<BR>

### Decoder

The role of Decoder is to take the latent space representation and reconstructs the original data from it. Inside the I included Attention Block and Residual Block, so encoder can also use it.
  ```py
    class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
  ```
  
Decoder takes outputs (Latent Representation) from the ==U-Net==, it transforms the latent representation back into an image. We convert the denoised latents generated by the reverse diffusion process into images using the VAE decoder.

During inference, we only need the VAE decoder to convert the denoised image into actual images.

  ```py
    nn.Conv2d(4, 4, kernel_size=1, padding=0),
  ```

The main purpose of this layer is to preserve the information. To ensure specific channels or features in the data remain unaltered and flow directly to subsequent layers. An ==identity mapping layer== allows this by passing the data through without any changes.

  ```py
  nn.Upsample(scale_factor=2),
  ```

Upsample is used for increase the spatial resolution of a tensor, typically referring to the height and width dimensions of an image or feature map.

  ```py
    nn.Conv2d(128, 3, kernel_size=3, padding=1),
  ```

This last layer is for converting 128 channels into 3 channels (RGB).

  ```py
    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x
  ```

  We need reverse the scaling before we send input into the decoder.


### Residual Block

This block help preserve information flow through the network by adding the input to the output of the output of a convolutional block.

  ```py
  class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)

        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)
  ```



### Understanding the Residual Block Architecture

<b> A series of layers </b>: Typically, these are convolutional layers that are responsible for extracting features from the input data. In most residual block implementations, there are two convolutional layers with a kernel size of 3x3 and padding of 1. The number of layers, kernel sizes, and other configurations can be adjusted depending on the specific application or problem being addressed.

The main purposes of using Residual Network:

- Addressing Vanishing Gradient Problem: Deep networks, like those often used in VAEs, can suffer from the vanishing gradient problem. This makes it difficult for the network to learn effectively in later layers. Residual blocks, with their skip connections, help alleviate this issue by allowing gradients to flow more easily through the network.
- Improved Learning Capacity: Residual networks significantly increase the learning capacity of the decoder. This allows it to capture the intricate relationships between the latent representation and the missing information in the original image.

Considerations:

- Computational Cost: Training deep residual networks can be more computationally expensive due to the increased number of layers. However, the benefits in reconstruction quality often outweigh the cost.
- Attention Mechanisms: Some VAE decoders might also use attention mechanisms alongside residual networks. These mechanisms can further focus the decoder on specific parts of the latent representation that are relevant for reconstruction.

When to use Residual Block?

- Residual blocks might be used in conjunction with layers that reshape tensors (like convolutional layers with strides for downsampling), the core purpose of the residual block itself is not related to the tensor shape changes. It's about facilitating better learning within the network.

techniques used for handling data shape changes in neural networks:

- Convolutional Layers: These layers can be used for downsampling (with strides) or upsampling (using techniques like transposed convolutions) to change the spatial dimensions of the data.
- Linear Layers: These layers are often used to project data from one feature space to another, potentially changing the number of channels in the tensor.
- Reshaping Operations: Operations like view or reshape can be used to explicitly change the dimensions of the tensor without applying any learnable parameters.


<b> Group normalization </b>: Group normalization is applied after each convolutional layer. It helps stabilize and accelerate training by normalizing the input to the following layer. It reduces the internal covariate shift, which occurs when the distribution of inputs to a layer changes during training.

- Group normalization divides channels into groups and normalizes the features within each group. It's computationally straightforward and doesn't have any restrictions regarding batch size. Group normalization performs particularly well in small batch scenarios where batch normalization suffers.

#### GroupNorm vs. BatchNorm

1. Performance in Large Batch Sizes: For tasks where large batch sizes are feasible, BatchNorm often outperforms GroupNorm. This is especially observed in tasks like image classification on large datasets. But BatchNorm’s performance degrades with smaller batch sizes because the estimates of mean and variance become less accurate.
2. Generalization in Different Scenarios: GroupNorm tends to generalize better in certain non-i.i.d settings (e.g., when the data distribution changes across batches because they may not be from the same distribution, i.e. not identically distributed). In the Stable Diffusion official repo, you can find that GroupNorm is adopted instead of BatchNorm.
3. Computational Overhead: BatchNorm might introduce a slight computational overhead during inference due to the need for maintaining and using running statistics. GroupNorm, on the other hand, doesn’t have this requirement.

<b>Non-linear activation functions</b> :

<p align="center"><img src="assets\SiLU vs. ReLU.png"></p>

- Advantages of SiLU over ReLU:

  - Gradient Flow: Unlike ReLU, which has a hard zero threshold, SiLU has a smooth, non-zero gradient for negative inputs. This smoother gradient allows for better backpropagation during training, potentially leading to faster convergence and avoiding the "dying ReLU" problem.
  - Zero-Centered Outputs: SiLU's output values tend to be centered around zero, which can simplify learning for subsequent layers in the network. ReLU outputs are always non-negative, which can introduce a bias towards positive values.

- Potential Benefits in VAEs:

  - VAEs often involve complex architectures with many layers. SiLU's smoother gradients and zero-centered outputs might contribute to more efficient training in such networks compared to ReLU.

- However, ReLU also has advantages:

  - Computational Efficiency: ReLU is a simpler function to compute compared to SiLU, which involves an exponential term. This can lead to faster training and inference.

## Attention Block

### Self-Attention

  ```py
    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
  ```

interim_shape is defined as a tuple with four dimensions: (batch_size, sequence_length, self.n_heads, self.d_head). This shape is chosen to facilitate the attention computation with multiple heads, where the attention is calculated within each head independently.

The view function is then applied to the q, k, and v tensors to reshape them into the interim_shape. The view function rearranges the tensor's dimensions without changing the underlying data. The total number of elements in the tensor must remain the same after reshaping.

  ```py
  weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output
  ```

<br>
<br>
<br>

## Clip Encoder

A CLIP encoder is a specific part of a Contrastive Language-Image Pre-training (CLIP) model. CLIP models are designed to understand the relationship between text and images. There are actually two encoders within a CLIP model:

- Text Encoder: This encoder takes text input, like a sentence or caption, and converts it into a numerical representation. This embedding captures the semantic meaning of the text.
- Image Encoder: This encoder takes an image as input and transforms it into another numerical representation. This embedding captures the visual content of the image.

However, in the Stable Diffusion primarily uses the CLIP text encoder. It focuses on using the encoded text description to refine noise into an image that aligns with the textual prompt.

  ```py
  class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding

        return x
  ```
CLIPEmbedding Class assigns a dense vector to each word using a pre-trained layer, capturing its meaning, and add position to the word, then combine both embeddings and positions together.

In essence, the position_embedding initialize as zeros and model doesn't update it directly within a single forward pass. Instead, the framework calculates gradients for all trainable parameters during backpropagation, and these gradients guide the updates to self.position_embedding over many training iterations.

  ```py
    class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        residue = x

        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension.

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x
  ```
Cliplayer has similar role to the encoder of transformer. It use self-Attetion to focus on important relationships between word in the text sequence. Using Feed-Forward Network to capture more complex aspects of the text's meaning by introducing non-linearity to the model. Futhermore, the reason why they normalize the data before attention block because they used technique called pre-norm Transformer which can sometimes achieve better performance compared to the original post-norm architecture.

<br>
<br>
<br>

## Diffusion Model (U-Net)

  ```py
    class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output
  ```

  ```py
    self.time_embedding = TimeEmbedding(320)
  ```
We give the U-Net not only the noisified image but also the time Step At which it was notified so the image the U-Net needs some way to understand this time step so this is why this time step which is a number will be converted into an embedding by using this layer. [Implementation](#TimeEmbedding-Class)

It just like the positional encoding of the Transformer model. This time tells the model at which step we arrived in the denoisification.

Diffusion Process:

- Stable diffusion works by gradually adding noise to an initial image (often pure noise) and then progressively removing it in a controlled manner.
- This noise removal process is guided by a series of steps, each with a specific noise level.

Time Embeddings as Step Information:

- Time embeddings are vector representations that encode information about the current step within the noise removal schedule.
- These embeddings are fed into the U-Net model alongside the image data itself.
- By incorporating this information, the U-Net can learn how to remove noise effectively based on the specific stage of the diffusion process.

Benefits of Time Embeddings:

- Improved Control: Time embeddings provide the model with more precise control over the noise removal process at each step.
- Stable Training: They can help the model learn a more stable and predictable way to remove noise, leading to better convergence during training.


   ```py
    def forward(self, latent, context, time):
   ```

- latent represents latent noise.
- context represents the conditioning information for image generation.
- time represents timestep.


<a id="TimeEmbedding-Class"></a>

### TimeEmbedding Class

  ```py
  class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        # (1, 1280) -> (1, 1280)
        x = F.silu(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x
  ```

For generating time embeddings, there are many approaches to implement it such as Linear Projection (We use this one), Sinusoidal Encodings, Learned Non-linear Embeddings, Recurrent Neural Networks (RNNs), and Rre-trained Embeddings. 

The Reasons for using this implementation:

-  Efficiency: Linear layers are computationally efficient, making this a suitable choice for real-time or resource-constrained applications. Since we use only 2 linear layers for increasing dimensionality, and using 1 non-linear for enhance the representation.
-  Control over Embedding Dimensionality: The choice of 4 * n_embd allows for increasing the embedding size compared to the input, potentially capturing more complex temporal information.


### UNET Class

The structure of the UNET is consist of Encoder, bottleneck, and Decoder:

#### Encoder

  ```py
        def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
  ```

- The encoders module list consists of several SwitchSequential blocks.
- These blocks contain either residual blocks or attention blocks, allowing the network to extract features from the input data at different resolutions.
- After each encoder block, the feature maps are downsampled (reduced in size) to capture higher-level features.
- The reasons that why we need to decrease the image size are:
  - It aims to extract features of increasing complexity, but the side effect is a decrease in image size. This happens because these operations combine neighboring pixels into a single value, effectively reducing the spatial resolution of the image.
  - Processing a smaller image requires fewer calculations compared to a larger one. This makes the network more efficient, especially when dealing with high-resolution images.

#### Bottleneck

  ```py
     self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
        )
  ```

- The bottleneck is a specific section in the U-Net where the image size reaches its minimum value.
- The bottleneck is a special block that processes the features from the final encoder.
- This design choice aims to capture the most critical and abstract features of the input image in a compressed representation within the bottleneck.

#### Decoder

  ```py
    self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])
  ```

- The decoders module list is similar to the encoders but in reverse order.
- It uses residual blocks, attention blocks, and upsampling (increases size) to reconstruct a detailed output.
- During decoding, skip connections from the encoders are concatenated with the decoder outputs, allowing the network to recover spatial information lost during downsampling.

#### What is SwitchSequential block?

  ```py
    class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
  ```

Similar to nn.Sequential, it takes an ordered list off layers as input during its initialization. During the forward pass, it iterates through this list, applying each layer's operation to the input data sequentially. It can recognize what are the parameters of each of them, and we'll apply accordingly.

#### Residual Block

For the Residual Block in the U-Net, it is similar to residual block in the VAE, but in this block they try to add the time embedding to the input.

#### Attention Block

  ```py
    class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.conv_output(x) + residue_long
  ```
  
Basically, we do layer normalize before sent the input into the self-attention block and do residual connection after it. Then, we gonna do similar thin again but in case, we use cross-attention block instead. After that, it go through feed forward layer which uses GeGLU function.

  ```py
    self.linear_geglu_1(x)
  ```

It uses to project the input features (x) to a higher dimension, essentially creating two separate feature spaces which are x and gate.

- gate: holds one of the projected feature spaces. It's passed through the GELU (Gaussian Error Linear Unit) activation function, which introduces a smooth, non-zero slope for negative inputs.
- x: holds another half of the projected feature spaces.

   ```py
    x = x * F.gelu(gate)
  ```
This part introduces non-linearity. The erf function ensures a smooth, non-zero slope for negative inputs. After this, using linear layer to projects output back to a suitable format for further processing.

So, It first projects the image data to a higher-dimensional space, allowing for more complex feature extraction. It then uses GELU to selectively emphasize specific features based on their characteristics. Finally, it modulates the original features (x) with these emphasized (gate) ones, potentially leading to a more robust representation of the objects.




#### GeGLU (Gated Linear Unit with GELU activation)

Mathematical Properties and Advantages

The nonlinear nature of GeGLU activations introduces a level of flexibility unseen in linear activation functions. This non-linearity allows neural networks to model complex, non-linear relationships in data, which is crucial for many real-world applications.

This characteristic enables deep learning models to approximate virtually any function, embodying the Universal Approximation Theorem.

Another advantage of GeGLU activations is their continuous differentiability. This property is important for gradient-based optimization algorithms, which are the backbone of most deep learning training processes. The smooth gradient of GeGLU activations can help these algorithms converge faster and more reliably.

When juxtaposed with activation stalwarts like ReLU, GELU, Sigmoid, and Swish, GeGLU shines in its ability to balance range, nonlinearity, and training efficiency. This balance is crucial for the robust performance of neural networks across a plethora of tasks, from image recognition to language processing. (Gated Linear Unit with GELU activation)

# Reference

### Residual Block

<https://medium.com/@neetu.sigger/a-comprehensive-guide-to-understanding-and-implementing-bottleneck-residual-blocks-6b420706f66b>
<https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c>



### Normalization

<https://www.linkedin.com/pulse/understanding-batch-normalization-layer-group-implementing-pasha-s>
<https://medium.com/@zljdanceholic/groupnorm-then-batchnorm-instancenorm-layernorm-e2b2a1d350a0>


### Clip Model

<https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab>

<https://medium.com/one-minute-machine-learning/clip-paper-explained-easily-in-3-levels-of-detail-61959814ad13>


### U-Net
<https://idiotdeveloper.com/attention-unet-and-its-implementation-in-tensorflow/>
#### interpolation and resampling: https://www.youtube.com/watch?v=rLMznzIslVA

Note: We generate noise in pipeline


VAEs work on learning how to compress the image and recover it. A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.

U-Nets work on analyzing images for segmentation and feature extraction while maintaining image size. The architecture of U-Net is unique in that it consists of a contracting path and an expansive path. The contracting path contains encoder layers that capture contextual information and reduce the spatial resolution of the input, while the expansive path contains decoder layers that decode the encoded data and use the information from the contracting path via skip connections to generate a segmentation map.

So VAE is the model that tries to compress the data while also preserving the features of the data. But U-Net focuses on analyzing and attending to the data while maintaining image size. 


### GeGLU
<https://medium.com/@juanc.olamendy/unlocking-the-power-of-geglu-advanced-activation-functions-in-deep-learning-444868d6d89c>
