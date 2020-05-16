from imutils import contours
import numpy as np
import imutils
import cv2
import myutils

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *int(i/3), 3 *int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

img=cv2.imread('mode.png')
cv_show('img',img)
#灰度图
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
#二值化
two=cv2.threshold(ref,200,255,cv2.THRESH_BINARY_INV)[1]
cv_show('two',two)
#去噪
kernel=np.ones((3,3),np.uint8)
res=cv2.morphologyEx(two,cv2.MORPH_OPEN,kernel)
cv_show('res',res)
compare=np.hstack((two,res))
#cv_show('compare',compare)
#blur=cv2.GaussianBlur(two,(5,5),1)
#cv_show('blur',blur) 去噪效果好但会模糊失真？

#计算轮廓
contours,hierarchy=cv2.findContours(res.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)

cv_show('img',img);
print(np.array(contours).shape)
contours=myutils.sort_contours(contours,method="left-to-right")[0]

#字典映射
digits={}

for(i,c) in enumerate(contours):
    (x,y,w,h)=cv2.boundingRect(c)
    roi=res[y:y+h,x:x+w]
    #cv_show('roi',roi)
    roi=cv2.resize(roi,(57,88))
    digits[i]=roi

#原图处理
image=cv2.imread('2.jpg')

image=myutils.resize(image,width=300)

size=image.shape
print(size)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)
ans=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ans',ans)

org_contours,org_hierarchy=cv2.findContours(ans.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

draw_img=image.copy();
cv2.drawContours(draw_img,org_contours,-1,(0,0,255),1)
cv_show('draw_img',draw_img)

soduko=np.zeros((9,9),np.int32)

origin=image.copy()

#提取符合的外界矩形
cnt=0  #已知数

for(i,c) in enumerate(org_contours):
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    roi = ans[y:y + h, x:x + w]

    if ar>0.3 and ar<0.9:
        if(w>5 and w<20) and (h>13and h<26):
            roi = ans[y:y + h, x:x + w]
            roi=cv2.resize(roi,(57,88))
            #cv_show('roi', roi)
            scores=[]
            for(digit,digitROI) in digits.items():
                result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
                (_, score, _, _)=cv2.minMaxLoc(result)
                scores.append(score)
            cnt=cnt+1
            print(str(np.argmax(scores)))
            print(np.argmax(scores))
            soduko[int(y*9/size[0])][int(x*9/size[1])]=np.argmax(scores)
            cv2.putText(origin,str(np.argmax(scores)),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#按一个顺序排序
print(cnt)
cv_show('origin',origin)
print(soduko)
#求解数独
solveSudoku(soduko)
print(soduko)
#填充图片
for i in range(9):
    for j in range(9):
        x=int((i+0.25)*size[1]/9)
        y=int((j+0.5)*size[0]/9)
        cv2.putText(image,str(soduko[j][i]),(x+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
print("\n验算:求每行每列的和\n")
row_sum=map(sum,soduko)
col_sum=map(sum,zip(*soduko))
print(list(row_sum))
print(list(col_sum))

cv_show('result',image)
answer=np.hstack((origin,image))
cv_show('res',answer)

