import cv2
import glob
# import ipyplot
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,10)

images_names = [img.split('/')[-1].split('.')[0] for img in glob.glob("./imagens_para_recorte/*")]
images = [np.asarray(Image.open(img)) for img in glob.glob("./imagens_para_recorte/*")]

# ipyplot.plot_images(images,img_width=300)

masks = [(img > 0).astype('f4') for img in images]
# ipyplot.plot_images(masks,img_width=300)

final_mask = np.clip(np.sum(np.array(masks, dtype='f4'),axis=0), 0, 1)

plt.imshow(final_mask)

cnts, _ = cv2.findContours(
    cv2.cvtColor((final_mask * 255).astype("u1"), cv2.COLOR_BGR2GRAY),
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_SIMPLE,
)[-2:]



draw = (final_mask * 255).astype("u1")
cv2.drawContours(draw, cnts, -1, (0,255,0), 3)
# plt.imshow(draw)

c = max(cnts, key=cv2.contourArea)


draw = (final_mask * 255).astype("u1")
cv2.drawContours(draw, [c], -1, (0,255,0), 5)
# plt.imshow(draw)

x, y, w, h = cv2.boundingRect(c)
x, y, w, h

croped = [img[y : y + h, x + 35 : x + w] for img in images]

# ipyplot.plot_images(croped,img_width=300)

for i, image in enumerate(croped):
    print(i)
    im = Image.fromarray(image)
    im.save(f'imagens_recortadas/{images_names[i]}.tiff')
