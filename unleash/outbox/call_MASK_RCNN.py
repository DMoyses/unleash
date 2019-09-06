
# coding: utf-8

# ## Run Object Detection
# 
# Container 2

# In[5]:


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[mask !=0] = [0,0,0]
    return image
                                                                                    
       


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
    image_copy = image

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
   

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        if label == ('person'):
            score = scores[i] if scores is not None else None
            if score > 0.73:
                caption = '{} {:.2f}'.format(label, score) if score else label
                mask = masks[:, :, i]

                image = apply_mask(image, mask, color)
                person_image = image_copy[y1:y2, x1:x2]
                mask = image[y1:y2, x1:x2]
                masked =cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
                person_image[masked !=0] = [255,255,255]
                
                
                #image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                #image = cv2.putText(
                #image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
                #)

    return image


# In[18]:


image_input = os.path.join(IMAGE_DIR,'pic1.jpg')
raw_image = skimage.io.imread(image_input)
image =cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
image_copy= image


results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
while True:
    cv2.imshow('picture',image)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break  
cv2.destroyAllWindows()  


# In[19]:


raw_image = skimage.io.imread(image_input)
imageo =cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
image_copy= imageo
boxes = r['rois']
y1, x1, y2, x2 = boxes[2]
person_image = image_copy[y1:y2, x1:x2]
masked_person_image = person_image
while True:
    cv2.imshow('picture',person_image)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break  
cv2.destroyAllWindows() 


# In[20]:


boxes = r['rois']
y1, x1, y2, x2 = boxes[2]
mask = image[y1:y2, x1:x2]
mask =cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
masked_person_image[mask !=0] = [255,255,255]
#masked_person_image =cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
#masked=cv2.cvtColor(masked_person_image, cv2.COLOR_RGB2GRAY)


# In[21]:


while True:
    cv2.imshow('picture',masked_person_image)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break  
cv2.destroyAllWindows() 


# In[22]:


person_pic = cv2.cvtColor(masked_person_image,cv2.COLOR_BGR2RGB)
skimage.io.imsave('person_pic.jpg',person_pic)


# In[23]:


width, height = person_pic.shape[:2]


# In[24]:


img = person_pic
img_hsv = cv2.cvtColor(person_pic, cv2.COLOR_RGB2HSV)
while True:
    cv2.imshow('picture',img)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break  
cv2.destroyAllWindows()


# In[25]:


# lower mask (0-10)
lower_red = np.array([0,100,100])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([150,100,100])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

while True:
    cv2.imshow('output_img',output_img)
    cv2.imshow('output_img',output_hsv)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break  
cv2.destroyAllWindows()


# In[26]:


mean_pic = np.mean(output_hsv)
mean_pic

