# Stable Diffusion

## Variational Autoencoder

## Encoder

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

## Decoder

The role of Decoder is to take the latent space representation and reconstructs the original data from it. Inside the I included Attention Block and Residual Block, so encoder can also use it.

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

  ```

### Understanding the Residual Block Architecture

<b> A series of layers </b>: Typically, these are convolutional layers that are responsible for extracting features from the input data. In most residual block implementations, there are two convolutional layers with a kernel size of 3x3 and padding of 1. The number of layers, kernel sizes, and other configurations can be adjusted depending on the specific application or problem being addressed.


<b> Group normalization </b>: Group normalization is applied after each convolutional layer. It helps stabilize and accelerate training by normalizing the input to the following layer. It reduces the internal covariate shift, which occurs when the distribution of inputs to a layer changes during training.

- Group normalization divides channels into groups and normalizes the features within each group. It's computationally straightforward and doesn't have any restrictions regarding batch size. Group normalization performs particularly well in small batch scenarios where batch normalization suffers.

### GroupNorm vs. BatchNorm:

1. Performance in Large Batch Sizes: For tasks where large batch sizes are feasible, BatchNorm often outperforms GroupNorm. This is especially observed in tasks like image classification on large datasets. But BatchNorm’s performance degrades with smaller batch sizes because the estimates of mean and variance become less accurate.
2. Generalization in Different Scenarios: GroupNorm tends to generalize better in certain non-i.i.d settings (e.g., when the data distribution changes across batches because they may not be from the same distribution, i.e. not identically distributed). In the Stable Diffusion official repo, you can find that GroupNorm is adopted instead of BatchNorm.
3. Computational Overhead: BatchNorm might introduce a slight computational overhead during inference due to the need for maintaining and using running statistics. GroupNorm, on the other hand, doesn’t have this requirement.






# Reference


## Residual Block
https://medium.com/@neetu.sigger/a-comprehensive-guide-to-understanding-and-implementing-bottleneck-residual-blocks-6b420706f66b
### Normalization
https://www.linkedin.com/pulse/understanding-batch-normalization-layer-group-implementing-pasha-s
https://medium.com/@zljdanceholic/groupnorm-then-batchnorm-instancenorm-layernorm-e2b2a1d350a0