import os
from PIL import Image
import os.path
import numpy as np





def create_database(files_dir):
   labels=[]
   images=[]
   for directory in sorted(os.listdir(files_dir)):
       for subdir in sorted(os.listdir(os.path.join(files_dir,directory))):

            images_data=[]
            
            for files in os.listdir(os.path.join(os.path.join(files_dir,directory),subdir)):
                if files_dir=='Emotion/':
                    with open(files_dir+directory+'/'+subdir+'/'+files) as f:

                        #database[files]=f.read().lstrip(" ").rstrip("\n")
                        p= int(f.read().lstrip(" ").rstrip("\n")[0])
                        #labels.append(int(f.read().lstrip(" ").rstrip("\n")[0]))
                        labels+=6*[p]
                else:
                    images_data.append(files)
            if files_dir == 'cohn-kanade-images/':        
                images_data.sort()
                image_list=images_data[-6:]
                for image in image_list:
                    im=Image.open(files_dir+directory+'/'+subdir+'/'+image).convert('L')
                    im=np.array(im.resize((100,100)))
                #print('Shape of images',im.shape)
                    images.append(im)
   if files_dir=='cohn-kanade-images/':
       return np.array(images)
   else:
       
       return np.array(labels).reshape(np.array(len(labels),1))
      
