import random, math
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
"""
Texture generation using Perlin noise
"""
class NoiseUtils:
    
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.gradientNumber = 256

        self.grid = [[]]
        self.gradients = []
        self.permutations = []
        self.img = np.empty((image_height, image_width))

        self.__generateGradientVectors()
        self.__normalizeGradientVectors()
        self.__generatePermutationsTable()

    def __generateGradientVectors(self):
        for i in range(self.gradientNumber):
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x * x + y * y < 1:
                    self.gradients.append([x, y])
                    break

    def __normalizeGradientVectors(self):
        for i in range(self.gradientNumber):
            x, y = self.gradients[i][0], self.gradients[i][1]
            length = math.sqrt(x * x + y * y)
            self.gradients[i] = [x / length, y / length]

    # The modern version of the Fisher-Yates shuffle
    def __generatePermutationsTable(self):
        self.permutations = [i for i in range(self.gradientNumber)]
        for i in reversed(range(self.gradientNumber)):
            j = random.randint(0, i)
            self.permutations[i], self.permutations[j] = \
                self.permutations[j], self.permutations[i]

    def getGradientIndex(self, x, y):
        return self.permutations[(x + self.permutations[y % self.gradientNumber]) % self.gradientNumber]

    def perlinNoise(self, x, y):
        qx0 = int(math.floor(x))
        qx1 = qx0 + 1

        qy0 = int(math.floor(y))
        qy1 = qy0 + 1

        q00 = self.getGradientIndex(qx0, qy0)
        q01 = self.getGradientIndex(qx1, qy0)
        q10 = self.getGradientIndex(qx0, qy1)
        q11 = self.getGradientIndex(qx1, qy1)

        tx0 = x - math.floor(x)
        tx1 = tx0 - 1

        ty0 = y - math.floor(y)
        ty1 = ty0 - 1

        v00 = self.gradients[q00][0] * tx0 + self.gradients[q00][1] * ty0
        v01 = self.gradients[q01][0] * tx1 + self.gradients[q01][1] * ty0
        v10 = self.gradients[q10][0] * tx0 + self.gradients[q10][1] * ty1
        v11 = self.gradients[q11][0] * tx1 + self.gradients[q11][1] * ty1

        wx = tx0 * tx0 * (3 - 2 * tx0)
        v0 = v00 + wx * (v01 - v00)
        v1 = v10 + wx * (v11 - v10)

        wy = ty0 * ty0 * (3 - 2 * ty0)
        return (v0 + wy * (v1 - v0)) * 0.5 + 1

    def makeTexture(self, texture = None):
        if texture is None:
            texture = self.cloud

        noise = {}
        max = min = None
        for i in range(self.image_height):
            for j in range(self.image_width):
                value = texture(i, j)
                noise[i, j] = value
                
                if max is None or max < value:
                    max = value

                if min is None or min > value:
                    min = value

        for i in range(self.image_height):
            for j in range(self.image_width):
                self.img[i, j] = (int) ((noise[i, j] - min) / (max - min) * 255)

    def fractalBrownianMotion(self, x, y, func):
        octaves = 12
        amplitude = 1.0
        frequency = 1.0 / self.image_width
        persistence = 0.5
        value = 0.0
        for k in range(octaves):
            value += func(x * frequency, y * frequency) * amplitude
            frequency *= 2
            amplitude *= persistence
        return value

    def cloud(self, x, y, func = None):
        if func is None:
            func = self.perlinNoise

        return self.fractalBrownianMotion(8 * x, 8 * y, func)

    def wood(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        frequency = 1.0 / self.image_width
        n = noise(4 * x * frequency, 4 * y * frequency) * 500
        return n - int(n)

    def marble(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        frequency = 1.0 / self.image_width
        n = self.fractalBrownianMotion(8 * x, 8 * y, self.perlinNoise)
        return (math.sin(10000 * x * frequency + 320 * (n - 0.5)) + 1) * 0.5


if __name__ == "__main__":
    image_height = 200
    image_width = 400
    noise = NoiseUtils(image_height, image_width)
    # noise is an instantiation of the class
    noise.makeTexture(texture = noise.marble)
    # calling the method 'makeTexture' - brackets follow

    # img = Image.new("L", (image_height, image_width))
    # pixels = img.load()

    noisy_img = noise.img
    # Accessing the'img' variable inside the class
    # noisy_img is the Perlin noise to add/multiply
    noisy_img = gaussian_filter(noisy_img, sigma=3)
    # gaussian filter changes min and max

    plt.imshow(noisy_img)
    plt.show()
    print(noisy_img.max(), noisy_img.min())
    noisy_img = noisy_img.astype('int')
    print(noisy_img.max(), noisy_img.min())
    noisy_img = Image.fromarray(noisy_img).convert("L")
    # turning the numpy array into an image class, pillows image class requires it to be [0, 255]

    # for i in range(0, image_height):
    #    for j in range(0, image_width):
    #         c = noise.img[i, j]
    #         pixels[i, j] = c
    #noisy_img.save("temp.png")