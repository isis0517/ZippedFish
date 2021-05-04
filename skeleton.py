import numpy as np
from numba import jit
import cv2
from scipy import signal
from skimage.measure import label
from skimage.measure import regionprops
import math


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def find_U(is_bound: np.array, points: np.ndarray, not_known: np.array) -> (np.array, int):
    """
    function : follow the paper, do the iterate of the U value

    parameters:
        in_bound : 2D-array:bool, the map of the point is boundary or not
        points : 1D-list:tuple , the list of points' coordinates. And it is sorted by the height of the points
        not_know : 2D-array:bool, the map of the point had pass or not

    
    return :
        U : 2D-array:int, the label number.
        count : int, the the maximum of label number.

    """

    row, col = is_bound.shape
    U = np.zeros((row, col))
    count = 1
    found = False
    for i in range(row):
        if found:
            break
        for j in range(col):
            if is_bound[i, j]:
                U[i, j] = count
                x, y = i, j
                found = True
                while found:
                    count += 1
                    is_bound[x, y] = False
                    found = False
                    if is_bound[x - 1, y + 1]:
                        U[x - 1, y + 1] = count
                        x, y = x - 1, y + 1
                        found = True

                    elif is_bound[x, y + 1]:
                        U[x, y + 1] = count
                        x, y = x, y + 1
                        found = True

                    elif is_bound[x + 1, y + 1]:
                        U[x + 1, y + 1] = count
                        x, y = x + 1, y + 1
                        found = True

                    elif is_bound[x + 1, y]:
                        U[x + 1, y] = count
                        x, y = x + 1, y
                        found = True

                    elif is_bound[x + 1, y - 1]:
                        U[x + 1, y - 1] = count
                        x, y = x + 1, y - 1
                        found = True

                    elif is_bound[x, y - 1]:
                        U[x, y - 1] = count
                        x, y = x, y - 1
                        found = True

                    elif is_bound[x - 1, y - 1]:
                        U[x - 1, y - 1] = count
                        x, y = x - 1, y - 1
                        found = True

                    elif is_bound[x - 1, y]:
                        U[x - 1, y] = count
                        x, y = x - 1, y
                        found = True
                found = True

    for n in range(points.shape[0]):
        i, j = points[n]
        temp0 = U[i, j]
        pos = i + 1, j
        if not_known[pos]:
            x, y = pos
            a = temp0
            num = 1
            umax = temp0
            umin = temp0

            if not not_known[x + 1, y]:
                temp = U[x + 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x, y + 1]:
                temp = U[x, y + 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x, y - 1]:
                temp = U[x, y - 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if umax - umin >= 2:
                U[pos] = temp0
            else:
                U[pos] = math.ceil(a / num)
            not_known[pos] = False

        pos = i, j + 1
        if not_known[pos]:
            x, y = pos
            a = temp0
            num = 1
            umax = temp0
            umin = temp0

            if not not_known[x + 1, y]:
                temp = U[x + 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x, y + 1]:
                temp = U[x, y + 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x - 1, y]:
                temp = U[x - 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1
            if umax - umin >= 2:
                U[pos] = temp0
            else:
                U[pos] = math.ceil(a / num)
            not_known[pos] = False

        pos = i - 1, j
        if not_known[pos]:
            x, y = pos
            a = temp0
            num = 1
            umax = temp0
            umin = temp0

            if not not_known[x, y + 1]:
                temp = U[x, y + 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x - 1, y]:
                temp = U[x - 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1
            if not not_known[x, y - 1]:
                temp = U[x, y - 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1
            if umax - umin >= 2:
                U[pos] = temp0
            else:
                U[pos] = math.ceil(a / num)
            not_known[pos] = False

        pos = i, j - 1
        if not_known[pos]:
            x, y = pos
            a = temp0
            num = 1
            umax = temp0
            umin = temp0

            if not not_known[x + 1, y]:
                temp = U[x + 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x - 1, y]:
                temp = U[x - 1, y]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1

            if not not_known[x, y - 1]:
                temp = U[x, y - 1]
                if umax < temp:
                    umax = temp
                if umin > temp:
                    umin = temp
                a += temp
                num += 1
            if umax - umin >= 2:
                U[pos] = temp0
            else:
                U[pos] = math.ceil(a / num)
            not_known[pos] = False

    return U, count


@jit(fastmath=True, nogil=True, cache=True)
def find_cen(U: np.array, count: int) -> np.array:
    """
    function:
        calculate the gradient fo the U and use threshold determined by count number to get sekelton img

    parameter:
        U : 2D-array:int, the label number.
        count : int, the the maximum of label number.
    return:
         : 2D-array:bool, the skeleton img
    """
    kerneldif = np.array([[0, 0, 0],
                          [0, 1, -1],
                          [0, 0, 0]])
    kerneldifT = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, -1, 0]])
    threshold = int(np.sqrt(count * 2))
    g1 = np.logical_or(np.abs(signal.convolve2d(U, kerneldif, mode='same')) > threshold,
                       np.abs(signal.convolve2d(U, kerneldifT, mode='same')) > threshold)

    U = U - int(count / 2)
    U = np.where(U < 0, U + count, U)

    g2 = np.logical_or(np.abs(signal.convolve2d(U, kerneldif, mode='same')) > threshold,
                       np.abs(signal.convolve2d(U, kerneldifT, mode='same')) > threshold)

    return np.logical_and(g1, g2)


def pre_img(img: np.array, **kwargs) -> np.array:
    """
    function :
        pre-prepare the img 
    """

    er_num = kwargs.setdefault('er_num', 1)
    di_num = kwargs.setdefault('di_num', 4)
    ekernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype='uint8')
    fkernel = np.ones((3, 3), dtype='uint8')

    img = cv2.erode(img, ekernel, iterations=er_num)
    img = cv2.dilate(img, fkernel, iterations=di_num)

    return img


@jit(fastmath=True, nogil=True, cache=True)
def find_cen2(U: np.array, count: int) -> np.array:
    """
    function:
        calculate the gradient fo the U and use threshold determined by count number to get skeleton img

    parameter:
        U : 2D-array:int, the label number.
        count : int, the the maximum of label number.
    return:
         : 2D-array:bool, the skeleton img
    """
    threshold = int(np.sqrt(count * 2))
    g1 = np.logical_or((np.abs(U[:, 1:] - U[:, :-1]) > threshold)[:-1, :],

                       (np.abs(U[:-1, :] - U[1:, :]) > threshold)[:, 1:])

    U = U - int(count / 2)
    U = np.where(U < 0, U + count, U)

    g2 = np.logical_or((np.abs(U[:, 1:] - U[:, :-1]) > threshold)[:-1, :],

                       (np.abs(U[:-1, :] - U[1:, :]) > threshold)[:, 1:])

    return np.pad(np.logical_and(g1, g2), ((1, 0), (1, 0)), 'constant', constant_values=(False, False))


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def find_ske(result: np.array, point: list, sub: np.array, sim_num: int = 10) -> np.ndarray:
    """
    function:
        from skeleton img take {sim_num} simples.

    parameter:
        result : 2D-array:float, the skeleton img
        point : list:tuple, the list of points' coordinates
        sub : 2D-array:bool, the origen image
    return:
        skeleton : 2D-array:float, the skeleton points coordinates
    """

    # this function use the idea of greedy. All point init with the farest distance(4*N) expect 
    # the origin(start) point is zero. function starts from the origin point and update the distance
    # of linking point if the distance is smaller than the point already had. the updated point will
    # be push into the queue. each time pop the point form the queue and update the distance until the 
    # queue is empty.

    # to a
    N = len(point)
    mapping = np.full(result.shape, 0)

    # mapping the position and the number
    num = 0
    for pos in point:
        mapping[pos] = num
        num += 1

        # cal the dis
    # start from the queue
    queue = [0] * 4 * N
    # all distance is 4*N(the farest)
    dis = np.full(N, 4. * N)
    # the queue parameter
    # the now is the index of queue's header
    now = 0
    # the tail is the index of queue's tail
    tail = 0

    # put the began point num
    queue[tail] = 0
    tail += 1
    dis[0] = 0  # update the distance

    while not tail == now:
        num = queue[now]
        now += 1

        i, j = point[num]
        l = dis[num]

        pos = (i + 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i + 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i - 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i + 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

    end1 = int(np.argmax(dis))
    pos = point[end1]
    x, y = pos

    # record the number of point near the endpoint.
    weight1 = 0
    for i in range(x - 10, x + 10):
        for j in range(y - 10, y + 10):
            if i >= sub.shape[0] or i < 0 or j >= sub.shape[1] or j < 0:
                continue
            if sub[i, j] > 100:
                weight1 += 1

    dis = np.full(N, 4. * N)
    now = 0
    tail = 0

    queue[tail] = end1
    tail += 1
    dis[end1] = 0

    while not tail == now:
        num = queue[now]
        now += 1

        i, j = point[num]
        l = dis[num]

        pos = (i + 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i + 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i - 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i + 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

    end2 = int(np.argmax(dis))
    pos = point[end2]
    x, y = pos

    weight2 = 0
    for i in range(x - 10, x + 10):
        for j in range(y - 10, y + 10):
            if i >= sub.shape[0] or i < 0 or j >= sub.shape[1] or j < 0:
                continue
            if sub[i, j] > 100:
                weight2 += 1

    # the endpoint have more nearby point is the head
    if weight1 > weight2:
        head = end1
        end = end2

    else:
        head = end2
        end = end1

    # cal the dis
    dis = np.full(N, 4. * N)
    now = 0
    tail = 0

    queue[tail] = head
    tail += 1
    dis[head] = 0

    while not tail == now:
        num = queue[now]
        now += 1

        i, j = point[num]
        l = dis[num]

        pos = (i + 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i + 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1
                queue[tail] = n
                tail += 1

        pos = (i - 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i + 1, j - 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

        pos = (i - 1, j + 1)
        if result[pos]:
            n = mapping[pos]
            if l + 1 < dis[n]:
                dis[n] = l + 1.41
                queue[tail] = n
                tail += 1

    fish_len = dis[end]
    align = np.argsort(dis)

    skeleton = [point[head]]

    dash = (fish_len + 1) / (sim_num - 1)
    step = dash

    for n in align:
        if dis[n] > step:
            skeleton.append(point[n])
            step += dash
    pos = point[end]
    skeleton.append(pos)
    return np.array(skeleton)


# 9/25更動
# connectivity = 1
# sub 用points 重建(避免其他label影響skeleton
# height 更簡短

def NT_skeleton(img, **kwargs):
    """

    :param img: the img of the zebrafish, it must be uint8 with 1 channel img
    :kwargs
        er_num: the number of iteration for erode.
        di_num: the number of iteration for dilate.
        sk_num: the number of node in skeleton.
    :return sk: The skeleton of fish
    """

    sk_num = kwargs.setdefault('sk_num', 10)
    ekernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype='uint8')
    assert img.dtype == 'uint8' or img.dtype == np.uint8, "it is not uint8"

    img = pre_img(img, **kwargs)
    labels = label(img, connectivity=1, background=0)
    group = regionprops(labels, cache=True)

    n, index = 0, 0
    area = 0
    for com in group:
        if com.area > area:
            index = n
            area = com.area
        n += 1
    com = group[index]

    # the minimums box contain all points of the img
    min_row, min_col, max_row, max_col = com.bbox
    # sub = img[min_row:max_row, min_col:max_col]

    min_row = min_row - 2
    min_col = min_col - 2
    max_row = max_row + 2
    max_col = max_col + 2

    # sub = np.pad(sub,((2,2),(2,2)),'constant',constant_values = (0,0))

    # the minimums box's size
    row, col = (max_row - min_row, max_col - min_col)
    points = com.coords - np.array([min_row, min_col])

    sub = np.zeros((row, col), dtype=np.uint8)
    sub[points[:, 0], points[:, 1]] = 255
    # cut to smallest size
    erosion = cv2.erode(sub, ekernel, iterations=1)
    height = cv2.distanceTransform(erosion, cv2.DIST_L2, 3)[points[:, 0], points[:, 1]]

    not_known = erosion > 100
    is_bound = np.logical_xor(erosion, sub)

    # points

    points_ord = np.argsort(height)

    U, count = find_U(is_bound, points[points_ord], not_known)  # 0.3ms

    result = find_cen(U, count)  # 1ms
    result = result * (erosion > 100)

    point = np.nonzero(result)
    point = list(zip(point[0], point[1]))

    skeleton = find_ske(result, point, sub, sim_num=sk_num) + np.array([min_row, min_col])

    return skeleton
