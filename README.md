# Maturaarbeit
This work follows from a paper written to qualify for university admission in Switzerland.
The main idea of this paper was to show that it is possible to apply Transfer Learning to GAN.
Therefore a GAN was first trained on the huge CelebA-Faces dataset (64x64 Pixel):

![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/GAN-TransferLearning/CelebA64_results.png)

CelebA64-GAN after 31 epochs of training on 276 images of Blondies:
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/GAN-TransferLearning/Blondies64_31.png)

CelebA32-GAN after 72 epochs of training on 276 images of Blondies:
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/GAN-TransferLearning/Blondies64_72.png)

CelebA32-GAN on 500 images of flowers (200 would also work fine) (Numbers denote relative order not epochs):
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/GAN-TransferLearning/Flower_evolution.png)
