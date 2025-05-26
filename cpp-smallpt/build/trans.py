import imageio
import os

for fname in os.listdir('.'):
    if fname.endswith('.ppm'):
        try:
            img = imageio.imread(fname)
            outname = fname.replace('.ppm', '.png')
            imageio.imwrite(outname, img)
        except Exception as e:
            print(f"Failed to convert {fname}: {e}")