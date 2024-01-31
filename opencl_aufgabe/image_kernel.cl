
__kernel void rgbToHsv(__global const uchar* inputImage, __global uchar* outputImage, 
    int width, int height, int depth)
{
    // store work-item's index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // check that the pixel doesn't go beyond image dimensions:
    if (x >= width || y >= height)
        return;

    // get the actual pixel location in the image
    const unsigned int loc = (y * width + x) * depth;
    outputImage[loc] = inputImage[loc];
    outputImage[loc + 1] = inputImage[loc + 1];
    outputImage[loc + 2] = inputImage[loc + 2];

    // HSV calculation
    float r = inputImage[loc] / 255.0f;
    float g = inputImage[loc + 1] / 255.0f;
    float b = inputImage[loc + 2] / 255.0f;

    float maxVal = max(r, max(g, b));
    float minVal = min(r, min(g, b));
    float diff = maxVal - minVal;
    float hue, saturation, value;
    value = maxVal;

    if (maxVal == minVal)
        hue = 0;
    else if (maxVal == r)
        hue = 60 * ((g - b) / diff);
    else if (maxVal == g)
        hue = 60 * ((b - r) / diff) + 120;
    else if (maxVal == b)
        hue = 60 * ((r - g) / diff) + 240;

    if (hue < 0)
        hue += 360;

    if (maxVal == 0)
        saturation = 0;
    else
        saturation = (diff / maxVal);

    // Modifying the output using hue, saturation, value
    outputImage[loc] = (uchar)hue / 2; 
    outputImage[loc + 1] = (uchar)(saturation * 255.0f);
    outputImage[loc + 2] = (uchar)(value * 255.0f);
}

__kernel void blur(__global uchar* inputImage, __global uchar* outputImage, const int width, 
    const int height, const int depth, const int kernelSize)
{
    // store work-item's index
    const int posx = get_global_id(0);
    const int posy = get_global_id(1);

    // Total number of pixels in the kernel
    int divider = ((2 * kernelSize + 1) * (2 * kernelSize + 1));

    // Blur operation
    for (int channels = 0; channels < depth; ++channels) { // Iterate over RGB channels
        float sum = 0.0f;

        for (int i = -kernelSize; i <= kernelSize; ++i) {
            for (int j = -kernelSize; j <= kernelSize; ++j) {
                // Make sure the indices are within bounds.
                int x = clamp(posx + i, 0, width - 1);
                int y = clamp(posy + j, 0, height - 1);

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    int index = (y * width + x) * depth + channels; // Calculate index for current channel
                    sum += inputImage[index];
                }
            }
        }

        float result = sum / divider;
        int outIndex = (posy * width + posx) * depth + channels; // Output index for current channel
        outputImage[outIndex] = (uchar)result;
    }
}