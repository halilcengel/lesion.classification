import skfmm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def grad(phi):
    Dx_n = phi - np.roll(phi, 1, axis=1)
    Dx_p = np.roll(phi, -1, axis=1) - phi
    Dy_n = phi - np.roll(phi, 1, axis=0)
    Dy_p = np.roll(phi, -1, axis=0) - phi
    Dx_0, Dy_0 = np.gradient(phi)
    return [Dx_0, Dx_n, Dx_p, Dy_0, Dy_n, Dy_p]


def curvature(phi):
    Dx_0, Dx_n, Dx_p, Dy_0, Dy_n, Dy_p = grad(phi)
    Nx = Dx_p / np.sqrt(1e-8 + Dx_p ** 2 + Dx_0 ** 2)
    Ny = Dy_p / np.sqrt(1e-8 + Dy_p ** 2 + Dy_0 ** 2)
    Dxx_n = grad(Nx)[1]
    Dyy_n = grad(Nx)[4]
    return Dxx_n + Dyy_n


def stopping_fun(x):
    fy, fx = np.gradient(x)
    norm = np.sqrt(fx ** 2 + fy ** 2)
    edge = 1. / (1. + norm ** 2)
    edge = cv2.GaussianBlur(edge, (5, 5), 5)
    edge[edge < 0.2] = 0
    return edge


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


def Levelset_move_curve(init_shape, iters, dv, I, V=""):
    h, w = init_shape.shape
    phies = []
    phi = skfmm.distance(init_shape)
    phies.append(phi)
    gI = stopping_fun(I)
    dt = 1
    final_segmentation = None

    for k in range(1, iters):
        Dx_0, Dx_n, Dx_p, Dy_0, Dy_n, Dy_p = grad(phi)

        max_Dx = np.maximum(Dx_n, -Dx_p, np.zeros((h, w)))
        max_Dy = np.maximum(Dy_n, -Dy_p, np.zeros((h, w)))
        grad_ = np.sqrt(max_Dx ** 2 + max_Dy ** 2)

        if (V == "gI"):
            phi = phi + dv * dt * (gI) * grad_
        elif (V == "Geo"):
            Nx = Dx_p / np.sqrt(1e-8 + Dx_p ** 2 + Dx_0 ** 2)
            Ny = Dy_p / np.sqrt(1e-8 + Dy_p ** 2 + Dy_0 ** 2)
            Dxx_n = Nx - np.roll(Nx, 1, axis=1)
            Dyy_n = Ny - np.roll(Ny, 1, axis=0)
            curv = Dxx_n + Dyy_n
            phi = phi + dt * (gI * (curv + dv) * grad_)

        init = np.zeros((h, w), dtype="uint8")
        init[phi < 0] = 255
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(init, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = np.zeros((output.shape))

        for i in range(0, nb_components):
            if sizes[i] > h * w / 100:
                img2[output == i + 1] = 255

        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
        final_segmentation = img2.copy()  # Store the final segmentation

        cv2.imshow("Evolution", img2)
        cv2.waitKey(5)

        img2[img2 == 0] = 1
        img2[img2 == 255] = -1

        if ((img2 > 0).all() or (img2 < 0).all()):
            break

        phi = skfmm.distance(img2)
        phies.append(phi)

        if len(phies) >= 2 and (np.linalg.norm(phies[-1] - phies[-2]) < 1e-3):
            break

    cv2.waitKey(0)

    if final_segmentation is not None:
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(I, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(final_segmentation, cmap='gray')
        plt.title('Final Segmentation')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        cv2.imwrite('final_segmentation.jpg', final_segmentation)

    return phies, final_segmentation


# Main execution
img = cv2.imread("cleaned_image.jpg", 0)
img = cv2.resize(img, (400, 400))
gray_scale = cv2.GaussianBlur(img, (5, 5), 1)
gray_scale = gray_scale - np.mean(gray_scale)
h, w = gray_scale.shape[:]

X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
init_shape = np.ones((h, w))
init_shape[5:h - 5, 5:w - 5] = -1

V = stopping_fun(gray_scale)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Velocity Field", V)
cv2.waitKey(0)

phies = Levelset_move_curve(init_shape, 100, 10, gray_scale, V="gI")