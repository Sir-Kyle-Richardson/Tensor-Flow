class ClassificationImage:
    standardShape = 28

    def __init__(self, nameOfImage):
        self.name = nameOfImage
        self.openImage()
        self.imagePreparation()

    def getImage(self):
        return self.img

    def openImage(self):
        try:
            self.img = Image.open(f"./images/{self.name}")
        except:
            print("Wrong file name")
            exit(1)

    def imagePreparation(self):
        (x, y) = self.img.size
        if x != y:
            max_variation = 0.1
            self.imageVariation(x, y)
        self.img.thumbnail(
            (ClassificationImage.standardShape, ClassificationImage.standardShape),
            Image.ANTIALIAS,
        )
        self.img = np.asarray(self.img)[:, :, 0]
        self.img = self.img / 255
        for i in range(0, ClassificationImage.standardShape):
            for j in range(ClassificationImage.standardShape):
                self.img[i, j] = abs(1 - self.img[i, j])
        self.plotMatrix()

    def imageVariation(self, x, y):
        max_variation = 0.1
        if abs(x - y) / x < max_variation and abs(x - y) / y < max_variation:
            if x > y:
                diff = int((x - y) / 2)
                self.img = self.img.crop((diff, 1, y + diff - 1, y))
            if y > x:
                diff = int((y - x) / 2)
                self.img = self.img.crop((1, diff, x, x + diff - 1))
        else:
            print(
                "The image is not a square (auto-correction for max 10% difference: x-y)"
            )

    def plotMatrix(self):
        fig = plt.figure()
        plt.imshow(self.img, cmap=plt.cm.binary)
        plt.colorbar()
        fig.savefig(fr"./28x28/{self.name}")
        plt.close(fig)
