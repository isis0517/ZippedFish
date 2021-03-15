import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
import time


def cut_fish_img(img, background, mean, std, mask, size):
    if img.ndim == 3 and img.shape[-1] == 3:
        a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        a = img
    a = cv2.absdiff(a, background)
    outer = a*mask
    outer = np.where(outer>(mean+std*10), 255, 0).astype('uint8')
    M = cv2.moments(outer)

    pos = (int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])) 

    return pos, cutting_img(img, pos, size)

def cutting_img(img, pos, size=150):
    cX, cY = pos
    dtype = img.dtype
    Len = size*2+1
    
    pad_arr=[[0,Len],[0,Len]]
    
    if img.ndim == 2: #1channel
        ndim = 1
        temp = np.zeros((Len, Len), dtype=dtype)
    elif img.shape[2] == 1:
        ndim = 1
        temp = np.zeros((Len, Len, 1), dtype=dtype)
    elif img.shape[2] == 3:
        ndim = 3
        temp = np.zeros((Len, Len, 3), dtype=dtype)
    else:
        raise Exception("wrong channel number!", img.shape)
    
    startX = cX - size
    endX = cX + size + 1
    
    if startX<0:
        pad_arr[0][0] = 0-startX
        startX=0
    if endX>img.shape[0]:
        pad_arr[0][1] = -endX+img.shape[0]
        endX=img.shape[0]
        
        
    startY = cY - size
    endY = cY + size + 1

    if startY<0:
        pad_arr[1][0] = 0-startY
        startY=0
    if endY>img.shape[1]:
        pad_arr[1][1] = -endY+img.shape[1]
        endY=img.shape[1]
    
    temp[pad_arr[0][0]:pad_arr[0][1], pad_arr[1][0]:pad_arr[1][1], ...]= img[startX:endX, startY:endY, ...]
    return temp

def mean_generater(imgs, background, mask):
    
    outer = []
    if imgs[0].ndim==3 and imgs[0].shape[-1]==3:
        for img in imgs:
            a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            a = cv2.absdiff(a, background)
            outer.append(a*mask)
    else:
        for img in imgs:
            a = cv2.absdiff(img, background)
            outer.append(a*mask)
        
    outer = np.array(outer)
    
    return np.mean(outer), np.std(outer)


def glue_img(img, pos, back):
    
    if not img.shape[0] == img.shape[1]:
        raise Exception("the img is not square!")
    
    size = int((img.shape[0]-1)/2)
    shape = back.shape[0:2]
    
    background = back.copy()
    
    startX = pos[0] - size
    endX = pos[0] + size + 1
    redsX, redeX = 0, int(2*size)+1
    
    if startX <= 0:
        redsX = -startX
        startX = 0
    
    if endX >= shape[0]:
        redeX = endX - shape[0]
        redeX = 2*size - redeX +1
        endX = shape[0]
    
    startY = pos[1] - size
    endY = pos[1] + size + 1
    redsY, redeY = 0, int(2*size)+1
    
    if startY <= 0:
        redsY = -startY
        startY = 0
    
    if endY >= shape[1]:
        redeY = endY - shape[1]
        redeY = 2*size - redeY +1
        endY = shape[1]
    
    #print(pos, (startY,endY), (redsY, redeY))
    
    background[startX:endX, startY:endY, ...] = img[redsX:redeX, redsY:redeY, ...]
    return background

def get_pos(img):#return the pos of the label with maximun area
    labels = label(img, connectivity=2, background=0)
    group = regionprops(labels, cache=True)
    area = 0
    pos = (0,0)
    for com in group:
        if com.area>area:
            area = com.area
            pos = com.centroid
    return  (int(pos[0]), int(pos[1]))

def optimizer(imgs, background, **kwargs):

    mask = kwargs.get('mask', np.full(imgs[0].shape, True, dtype=np.bool))
    size = kwargs.get('size', 100)
    op_lost = kwargs.get('op_lost', 1)
    thres = kwargs.get('thres', 10)
    mean = kwargs.get('mean', -1)
    std = kwargs.get('std', -1)

    if mean*std < 0 :
        raise Exception("you should enter mean and std")
    elif mean < 0:
        mean, std = mean_generater(imgs, background, mask)
    #print((outer_mean, outer_std),(inner_mean, inner_std))
    
    gimgs = []
    
    if imgs[0].ndim == 3 and imgs.shape[-1] == 3:
        for img in imgs:
            gimgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        gimgs = imgs
        
    N = len(gimgs)
    is_pass = False
    op_size = size  
    
    #find the average point numbers in img and its' std
    
    ori_nums = []
    imgs_thre = []
    poses = []
    
    for img in gimgs:
        
        ori_num = 0
        a = cv2.absdiff(img, background)

        img = a*mask
        img = img > (mean + std*thres)
        pos = get_pos(img)
        poses.append(pos)
        imgs_thre.append(img.copy())
        ori_num += np.sum(img*1)
        
        ori_nums.append(ori_num)
    
    plt.plot(ori_nums)
    plt.show()
    ori_mean = np.mean(ori_nums)
    ori_std = np.std(ori_nums)

    print(f"the average number is {ori_mean} per img with std = {ori_std}")
    
    size = int(np.sqrt(ori_mean*2))
    
    for s in range(10):
        pos_err = 0
        com_nums = []
        for s in range(len(gimgs)):
            
            #cutting img
            img = cutting_img(imgs_thre[s], poses[s], size)
            pos = get_pos(img)
            pos_err += np.sqrt((pos[0]-size)**2+(pos[1]-size)**2)

            #calculate the point we get after glue back(decompressed)
            com_num = 0
            com_num += np.sum(img*1)
            com_nums.append(com_num)
        
        com_mean = np.mean(com_nums)
        com_std = np.std(com_nums)
        
        if pos_err < 1 :
            if not is_pass:
                op_size = size
            is_pass = True
            
            if op_size>size:
                op_size = size
            size = size*0.8
        
        if not is_pass:
            size = size*1.5
        
        else:
            size *= 1.1
        
        print(f"{s+1}th op_size:{op_size} ||@size={size}, pos_err={pos_err}, mean={com_mean}, std={com_std}")
        size = int(size)
    
    return int(op_size*1.1)
        


# ### half pos

# In[5]:


def cutting_img_LW(img, pos, L, W):
    cX, cY = pos
    dtype = img.dtype
    
    L = 2*(L//2)+1
    W = 2*(W//2)+1
    
    pad_arr=[[0,L],[0,W]]
    
    if img.ndim == 2: #1channel
        ndim = 1
        temp = np.zeros((L, W), dtype=dtype)
    elif img.shape[2] == 1:
        ndim = 1
        temp = np.zeros((L, W, 1), dtype=dtype)
    elif img.shape[2] == 3:
        ndim = 3
        temp = np.zeros((L, W, 3), dtype=dtype)
    else:
        raise Exception("wrong channel number!", img.shape)
    
    startX = cX - L//2
    endX = cX + L//2 + 1
    
    if startX<0:
        pad_arr[0][0] = 0-startX
        startX=0
    if endX>img.shape[0]:
        pad_arr[0][1] = -endX+img.shape[0]
        endX=img.shape[0]
        
        
    startY = cY - W//2
    endY = cY + W//2 + 1

    if startY<0:
        pad_arr[1][0] = 0-startY
        startY=0
    if endY>img.shape[1]:
        pad_arr[1][1] = -endY+img.shape[1]
        endY=img.shape[1]
    
    temp[pad_arr[0][0]:pad_arr[0][1], pad_arr[1][0]:pad_arr[1][1], ...]= img[startX:endX, startY:endY, ...]
    return temp, (cX - L//2, cY - W//2)

def get_pos_size(img,size):#return the pos of the label with maximun area
    M = cv2.moments(img)

    cX = int(M["m01"] / M["m00"])
    cY = int(M["m10"] / M["m00"])
    
    if img.shape[0] > size*2:
        temp, red = cutting_img_LW(img, (cX,cY), img.shape[0]//2, img.shape[1])
        pos = get_pos_size(temp, size)
        return (pos[0]+red[0], pos[1]+red[1])
    if img.shape[1] > size*2:
        temp, red = cutting_img_LW(img, (cX,cY), img.shape[0], img.shape[1]//2)
        pos = get_pos_size(temp, size)
        return (pos[0]+red[0], pos[1]+red[1])
    
    return (cX, cY)




class BG_tiff: 
    def __init__(self, imgslist, nbckgnd = 1000):
        
        self.cir_num = -1
        self.nbckgnd = nbckgnd
        self.imgslist = imgslist
        timg = tiff.TiffFile(imgslist[0]).asarray()
        if timg.ndim == 2 or timg.shape[-1] == 1:
            self.mono = True
        else:
            self.mono = False
            
        if self.mono:
            tnum = np.random.randint(len(imgslist),size=nbckgnd)
            tlist=[]
            for s in tnum:
                tlist.append(imgslist[s])
            
            img_shape = tiff.imread(imgslist[0]).shape[0:2]
            total = np.zeros(img_shape)
            for file in tlist:
                b = tiff.TiffFile(file).asarray()
                total = total + b 
            self.background = (total/nbckgnd).astype('uint8')
            self.cbackground = self.background
            self.shape = img_shape
            
        else:
            tnum = np.random.randint(len(imgslist),size=nbckgnd)
            tlist=[]
            for s in tnum:
                tlist.append(imgslist[s])

            img_shape = tiff.imread(imgslist[0]).shape[0:2]
            total = np.zeros(img_shape)
            ctotal = np.zeros((*img_shape,3))
            for file in tlist:
                b = tiff.TiffFile(file).asarray()
                ctotal += b
                b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                total = total + b 
            self.cbackground = (ctotal/nbckgnd).astype('uint8')
            self.background = (total/nbckgnd).astype('uint8')
            self.shape = img_shape
            
    def circlelize(self):
        if self.cir_num > 0:
            return
        back_itr = np.zeros(self.shape, dtype='uint8')
        back_itr = cv2.normalize(self.background.copy(), back_itr, 0, 255,  norm_type = cv2.NORM_MINMAX)
   
        R = self.shape[0]
        pos = [0]*5
        R_L = [0]*5
        cir_num = 0
        while True:
            try:
                pos[cir_num], R_L[cir_num] = self.findcir(back_itr, R-50)
            except:
                break
            R = R_L[cir_num]

            mask = np.zeros(back_itr.shape, dtype = 'uint8')
            mask = cv2.circle(mask, pos[cir_num], R_L[cir_num]-2, 255, -1)>100

            back_itr = back_itr*mask
            back_itr = cv2.normalize(back_itr, back_itr, 0, 255,  norm_type = cv2.NORM_MINMAX)
            
            cir_num+=1
            
        self.cir_num = cir_num
        self.R_L = R_L
        self.pos = pos
        
    def findcir(self, img, maxR):
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, maxRadius = int(maxR))
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        
        for (x,y,r) in circles[:1]:
            pos = (x, y)
            MaxR = r

        return pos, MaxR
    
    def gen_cirmask(self, Rn, ch=1):
        self.circlelize()
        if Rn > self.cir_num:
            raise Exception("GG, too big")
            return
        
        if ch == 3:
            shape = (*(self.shape),3)
            black = (0,0,0)
            white = (255,255,255)
        elif ch == 1:
            shape = self.shape
            black = (0)
            white = 255
            
        if Rn == 0:
            mask = np.full(shape, 255, dtype = 'uint8')
            mask = cv2.circle(mask, self.pos[0], self.R_L[0], black, -1)
            mask = mask>100
            
        elif Rn == self.cir_num:
            mask = np.full(shape, 0, dtype = 'uint8')
            mask = cv2.circle(mask, self.pos[Rn-1], self.R_L[Rn-1], white, -1)
            mask = mask>100
        
        else:
            mask = np.zeros(shape, dtype = 'uint8')
            cv2.circle(mask, self.pos[Rn-1], self.R_L[Rn-1], white, -1)
            cv2.circle(mask, self.pos[Rn], self.R_L[Rn], black, -1)
            mask = mask>100
            
        return mask
    
    def auto_mask(self):
        
        tnum = np.random.randint(len(self.imgslist),size=int(self.nbckgnd*0.1))
        test=[]
        for s in tnum:
            img = tiff.TiffFile(self.imgslist[s]).asarray()
            if not self.mono:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            a = cv2.absdiff(img, self.background)
            test.append(a.copy())
        
        mean = np.mean(test)
        std = np.std(test)
        
        test_imgs = []
        for a in test:
            img = a>mean+10*std
            test_imgs.append(img)
        
        self.circlelize()
        opt_pdiff = np.inf
        opt_Rn = 0
        for s in range(self.cir_num-1):
            outer_mask = self.gen_cirmask(s)
            inner_mask = self.gen_cirmask(s+1)
            pdiff = 0
            
            for img in test_imgs:
                a = np.sum(img*outer_mask)
                b = np.sum(img*inner_mask)
                if a == b:
                    a = a+0.1
                pdiff += (a+b)/(a-b)
            if pdiff < opt_pdiff:
                opt_Rn = s
                opt_pdiff = pdiff
        
        return opt_Rn
        

