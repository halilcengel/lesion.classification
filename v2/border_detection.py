# Programa que detecta los bordes en una imagen
# Program that detects the edges in an image
# =====================================================#
from lib2to3.pytree import convert
import sys
from PIL import Image, ImageFilter

#  Carga de imagen /Upload the image
image = Image.open("watershed_segments/ISIC_0000042_mask.jpg").convert('L')


def detector_debordes(tipo):
    if tipo == 'Prewitt':
        factor = 6
        coeficientes_h = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        coeficientes_v = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        coeficientes_h1 = [1, 0, -1, 2, 0, -2, 1, 0, -1]
        coeficientes_v1 = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    elif tipo == 'Sobel':
        factor = 8
        coeficientes_h = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
        coeficientes_v = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
        coeficientes_h1 = [1, 0, -1, 2, 0, -2, 1, 0, -1]
        coeficientes_v1 = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    else:
        sys.exit(0)
    datos_h = image.filter(ImageFilter.Kernel((3, 3), coeficientes_h, factor)).getdata()
    datos_v = image.filter(ImageFilter.Kernel((3, 3), coeficientes_v, factor)).getdata()
    datos = []

    for x in range(len(datos_h)):
        datos.append(round(((datos_h[x] ** 2) + (datos_v[x] ** 2)) ** 0.5))

    datos_h = image.filter(ImageFilter.Kernel((3, 3), coeficientes_h, factor)).getdata()
    datos_v = image.filter(ImageFilter.Kernel((3, 3), coeficientes_v, factor)).getdata()

    datos_signo_contrario = []

    for x in range(len(datos_h)):
        datos_signo_contrario.append(round(((datos_h[x] ** 2) + (datos_v[x] ** 2)) ** 0.5))

    datos_bordes = []

    for x in range(len(datos_h)):
        datos_bordes.append(datos[x] + datos_signo_contrario[x])

    return datos_bordes


datos_bordes = detector_debordes('Prewitt')

nueva_imagen = Image.new('L', image.size)
nueva_imagen.putdata(datos_bordes)
nueva_imagen.save('tortu_N.jpg')
image.close()
nueva_imagen.close()