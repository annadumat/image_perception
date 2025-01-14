from PIL import Image, ImageEnhance
import os

#create contrast out of category files with images

path = r"C:\Users\Nutzer\Desktop\Stimuli\Menschen"
contrasts = [3,5,10,15,25]

for filename in os.listdir(path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
         image = (os.path.join(path, filename))
         print(image)

         # read the image
         im = Image.open(image)

         # image brightness enhancer
         enhancer = ImageEnhance.Contrast(im)
         ordnerpath= os.path.join(path, filename+"_")
         os.mkdir(ordnerpath)

         for i in contrasts:
             factor = i/10  # decrease constrast
             im_output = enhancer.enhance(factor)
             im_output.save(os.path.join(ordnerpath, "CONTRAST_"+str(i/10)+"_"+filename))



    else:
        continue



