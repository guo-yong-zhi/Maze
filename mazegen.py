# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:28:52 2017

@author: momos
"""

import random
import numpy as np
from matplotlib import pyplot as plt

array = np.array

from graph import graph, spanningTree_BrkCir_BFS
import graphHelper as gh

from graph import spanningTree_path_len


class maze:
    def __init__(self, mask, start=None, end=None, length=None):
        """
        从图片mask（二维0-1 ndarray）生成迷宫，图像需连续，0为透明
        一个像素代表迷宫中一个格子
        start、end都是tuple(row,col)坐标，默认为最上左角、最下右角
        """
        self.mask = mask
        self.maskrows, self.maskcols = self.maskshape = mask.shape[:2]
        self.length = length or 2 * (self.maskrows + self.maskcols)

        # 每个像素的id
        self.idmz = idmz = gh.to_id_image(mask, transOld=0, transNew=-1)

        # 相邻像素的对儿
        self.nepairs = gh.neiPairsOf(idmz, trans=-1, shift=gh.shift_neonespadding)

        # 对应的图
        self.G = G = graph(self.nepairs)
        #    print(idmz[start],idmz[end])
        #    print(G.search(idmz[start],idmz[end]))

        self.start_id = min(G.vertices) if start is None else idmz[start]
        self.end_id = max(G.vertices) if end is None else idmz[end]
        self.start_coord, self.end_coord = gh.toCoords(
            [self.start_id, self.end_id], self.maskshape)
        # G的最小生成树
        self.T = T = spanningTree_path_len(G, self.start_id, self.end_id, self.length)

        # 迷宫的矩阵表示，M.shape = (self.maskrows, self.maskcols, 4)
        self.M = MOfT(T, self.maskshape)

    def gen_image(self, pathwidth=6, wallwidth=3, RGBA=False):
        im_ma = imageOfM(self.M, pathwidth=pathwidth, wallwidth=wallwidth,
                         mask=self.mask)
        self.image, self.image_mask = im_ma

        pw, ww = pathwidth, wallwidth
        unit = pw + ww
        cell_w = 2 * ww + pw
        self.toImageCoord = lambda c: c * unit + cell_w // 2
        self.image_start_coord = self.toImageCoord(self.start_coord)
        self.image_end_coord = self.toImageCoord(self.end_coord)

        if RGBA:
            return im2RGBA(self.image, self.image_mask)
        else:
            return self.image


def mazeGen_rect(num_rows=20, num_cols=20, start=None, end=None,
                 length=None):
    """
    生成矩形迷宫
    """
    mz = np.ones((num_rows, num_cols), np.int8)
    return maze(mz, start, end, length)


def mazeGen_BrkCir(num_rows=20, num_cols=20):
    """
    随机生成矩形迷宫，不能指定path长度
    """
    mz = np.ones((num_rows, num_cols), np.int8)
    idmz = gh.to_id_image(mz)
    nepairs = gh.neiPairsOf(idmz, shift=gh.shift_neonespadding)
    G = graph(nepairs)
    T = spanningTree_BrkCir_BFS(G)
    return T


def MOfT(T, shape):
    """
    由最小生成树产生迷宫矩阵M
    # The array M is going to hold the array information for each cell.  
    # The four coordinates tell if walls exist on those sides    
    # M(LEFT, UP, RIGHT, DOWN)  
    """
    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
    M = np.zeros((*shape, 4), np.int8)
    id2c = lambda i: gh.toCoords(i, shape)
    memmap = {(0, -1): LEFT, (-1, 0): UP, (0, 1): RIGHT, (1, 0): DOWN}
    for v1, v2 in T.edges:
        c1, c2 = id2c([v1, v2])
        delta = c2 - c1
        ind = memmap[(*delta,)]
        M[(*c1, ind)] = 1
        ind2 = memmap[(*(-delta),)]
        M[(*c2, ind2)] = 1
    return M


def imageOfM(M, mask=None, pathwidth=1, wallwidth=1):
    """
    mask: An array has the same shape as M, consists of 0, 1 (0 for transparent).
    return: image when mask is None,  otherwise (image, image mask) tuple.
    (the return image is image of maze)
    """
    pw, ww = pathwidth, wallwidth
    unit = pw + ww
    r_m, c_m = M.shape[:2]
    r_im = r_m * unit + ww
    c_im = c_m * unit + ww

    im = np.ones((r_im, c_im), np.int8)
    if mask is not None:
        im_mask = np.zeros((r_im, c_im), np.int8)

    cell_w = 2 * ww + pw

    def paintCell(cell, code):
        code = code.copy()
        code[code > 0] = 1
        code[code <= 0] = 0
        cell[ww:ww + pw, ww:ww + pw] = 1
        cell[0:ww, 0:ww] = 0
        cell[0:ww, -ww:] = 0
        cell[-ww:, 0:ww] = 0
        cell[-ww:, -ww:] = 0
        LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
        cell[ww:ww + pw, 0:ww] = code[LEFT]
        cell[0:ww, ww:ww + pw] = code[UP]
        cell[ww:ww + pw, -ww:] = code[RIGHT]
        cell[-ww:, ww:ww + pw] = code[DOWN]

    for r in range(r_m):
        for c in range(c_m):
            if mask is not None and mask[r, c] > 0:
                im_mask[r * unit:r * unit + cell_w, c * unit:c * unit + cell_w] = 1
            if mask is None or mask[r, c] > 0:
                wallCode = M[r, c]
                cell = im[r * unit:r * unit + cell_w, c * unit:c * unit + cell_w]
                paintCell(cell, wallCode)
    import matplotlib.cm as cm
    # Display the image  
    plt.imshow(im, cmap=cm.Greys_r, interpolation='none')
    plt.show()

    if mask is not None:
        return im, im_mask
    else:
        return im


def im2RGBA(im, mask=None):
    palette = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], np.uint8)
    im = palette[im]
    if mask is not None:
        im[mask <= 0, -1] = 0
    return im


def standardize(im):
    sim = np.roll(im, 1, axis=0)
    mask = (sim == im).all(axis=1)
    mask[0] = False
    im = im[~mask]

    sim = np.roll(im, 1, axis=1)
    mask = (sim == im).all(axis=0)
    mask[0] = False
    im = im[:, ~mask]
    return im


# %%
if __name__ == "__main__":
    shape = (30, 30)
    #    t = mazeGen_with_len(*shape)#,start=(0,0),end=(0,1))
    mi = np.zeros(shape)
    mi[10:20] = 1
    mi[:, 10:20] = 1
    #    t = mazeGen_from_im(mi)
    #    m = MOfT(t, shape)
    #    im,mask = imageOfM(m, pathwidth=6, wallwidth=3, mask=mi)

    mz = maze(mi)
    im = mz.gen_image(RGBA=True)
    plt.imsave("maze.png", im)
    # %%
    #    stdmaze = standardize(im)
    stdmaze = mz.gen_image(1, 1)
    idim = gh.to_id_image(stdmaze)
    nepairs = gh.neiPairsOf(idim, shift=gh.shift_neonespadding)
    g = graph(nepairs)

    way = g.search(gh.toIDs(mz.image_start_coord, stdmaze.shape),
                   gh.toIDs(mz.image_end_coord, stdmaze.shape))
    start_end = np.c_[mz.image_start_coord, mz.image_end_coord][::-1, :]

    fig, ax = plt.subplots()
    gh.plotGroup(ax, stdmaze, way)
    ax.plot(*start_end, 'o')
    print("the length of the way is", len(way))
