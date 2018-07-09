#!/usr/bin/env python3.6 
"""
//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of load
//        bit 31~0 - load[31:0] (Read/Write)
// 0x14 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
"""

from pynq import Overlay
from pynq.lib.dma import DMA
from pynq import Xlnk
from pynq import MMIO

import numpy as np
import os

import log_initializer
from logging import getLogger, NullHandler, DEBUG, INFO

logger = getLogger(__name__)
logger.addHandler(NullHandler())
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)


class MaxLayer(object):
    def __init__(self, layer, fm, dim, xlnk, batchsize=1):
        self.layer = layer
        self.fm = fm
        self.dim = dim
        self.xlnk = xlnk
        self.batchsize = batchsize

        self.ol = Overlay(os.path.dirname(os.path.realpath(__file__)) + "/bitstream/" + self.layer + ".bit")
        self.dma = self.ol.axi_dma_0

        self.cmaOutMax = []
        for b in range(self.batchsize):
            self.cmaOutMax.append(self.xlnk.cma_array(shape=(self.fm * (self.dim ** 2),), dtype=np.float32))


    def __call__(self, input):
        self.ol.download()

        self.dma.sendchannel.start()
        self.dma.recvchannel.start()

        b = 0
        while b < self.batchsize:
            self.dma.recvchannel.transfer(self.cmaOutMax[b])
            self.dma.sendchannel.transfer(input[b])

            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()

            b += 1

        return self.cmaOutMax


class ConvLayer(object):
    def __init__(self, layer, fm, dim, xlnk, runFactor=1, batchsize=1):
        self.layer = layer
        self.fm = fm
        self.dim = dim
        self.xlnk = xlnk
        self.runFactor = runFactor
        self.batchsize = batchsize

        self.COMPUTE = 0
        self.CONV_WEIGHT = 1

        self.ol = Overlay(os.path.dirname(os.path.realpath(__file__)) + "/bitstream/" + layer + ".bit")
        self.dma = self.ol.axi_dma_0
        self.ip = MMIO(self.ol.ip_dict[self.layer]['phys_addr'], self.ol.ip_dict[self.layer]['addr_range'])

        self.wBuff = []
        self.initWeights()

        self.cmaOut = []
        self.cmaTemp = []
        for b in range(self.batchsize):
            self.cmaOut.append(self.xlnk.cma_array(shape=(self.fm * (self.dim ** 2),), dtype=np.float32))
            self.allocaCmaTemp()


    def __call__(self, input):
        self.ol.download()

        self.dma.sendchannel.start()
        self.dma.recvchannel.start()

        full = [list() for b in range(self.batchsize)]

        r = 0
        while r < self.runFactor:
            self.ip.write(0x10, self.CONV_WEIGHT)
            self.ip.write(0x00, 1)  # ap_start

            self.dma.sendchannel.transfer(self.wBuff[r])
            self.dma.sendchannel.wait()

            b = 0
            while b < self.batchsize:
                self.ip.write(0x10, self.COMPUTE)

                self.ip.write(0x00, 1)  # ap_start

                self.dma.recvchannel.transfer(self.cmaTemp[b][r])
                self.dma.sendchannel.transfer(input[b])

                self.dma.sendchannel.wait()
                self.dma.recvchannel.wait()

                temp = self.cmaTemp[b][r].reshape((self.dim**2, int(self.fm/self.runFactor)))
                temp = temp.transpose()
                temp = temp.reshape(int(self.fm/self.runFactor)*(self.dim**2))
                full[b] = np.concatenate((full[b], temp))

                b += 1

            r += 1

        b = 0
        while b < self.batchsize:
            full[b] = full[b].reshape(self.fm, (self.dim**2)).transpose().flatten()
            np.copyto(self.cmaOut[b], full[b])
            b +=1

        return self.cmaOut


    def allocaCmaTemp(self):
        t = []
        for i in range(self.runFactor):
            t.append(self.xlnk.cma_array(shape=(int(self.fm/self.runFactor) * (self.dim ** 2),), dtype=np.float32))
        self.cmaTemp.append(t)


    def initWeights(self):
        w = np.load(os.path.dirname(os.path.realpath(__file__)) + "/weights/" + self.layer + "/W.npy")
        b = np.load(os.path.dirname(os.path.realpath(__file__)) + "/weights/" + self.layer + "/b.npy")

        w = w.reshape((self.runFactor, -1))
        b = b.reshape((self.runFactor, -1))

        for i in range(self.runFactor):
            buff = self.xlnk.cma_array(shape=(w[i].size + b[i].size,), dtype=np.float32)
            np.concatenate((w[i], b[i]), out=buff)
            self.wBuff.append(buff)


class CnnFPGA(object):
    def __init__(self, batchsize=8):
        logger.info('Define a HyperFace CNN model on FPGA')

        self.batchsize = batchsize
        self.batchsizeMax = batchsize

        self.xlnk = Xlnk()

        self.runFactor = {'conv_1': 1, 'conv_1a': 8, 'conv_2': 8, 'conv_3': 16,
                          'conv_3a': 8, 'conv_4': 16, 'conv_5': 16, 'conv_all': 2}

        self.cmaIn = []
        self.inConvAll = []
        b = 0
        while b < self.batchsize:
            self.cmaIn.append(self.xlnk.cma_array(shape=(3*227*227,), dtype=np.float32))
            self.inConvAll.append(self.xlnk.cma_array(shape=(768*6*6,), dtype=np.float32))
            b += 1

        self.conv1 = ConvLayer('conv_1', 96, 55, self.xlnk, runFactor=1, batchsize=self.batchsize)
        self.max1 = MaxLayer('max_1', 96, 27, self.xlnk, batchsize=self.batchsize)
        self.conv1a = ConvLayer('conv_1a', 256, 6, self.xlnk, runFactor=8, batchsize=self.batchsize)

        self.conv2 = ConvLayer('conv_2', 256, 27, self.xlnk, runFactor=8, batchsize = self.batchsize)
        self.max2 = MaxLayer('max_2', 256, 13, self.xlnk, batchsize=self.batchsize)
        self.conv3 = ConvLayer('conv_3', 384, 13, self.xlnk, runFactor=16, batchsize = self.batchsize)
        self.conv3a = ConvLayer('conv_3a', 256, 6, self.xlnk, runFactor=8, batchsize = self.batchsize)

        self.conv4 = ConvLayer('conv_4', 384, 13, self.xlnk, runFactor=16, batchsize = self.batchsize)
        self.conv5 = ConvLayer('conv_5', 256, 13, self.xlnk, runFactor=16, batchsize = self.batchsize)
        self.max5 = MaxLayer('max_5', 256, 6, self.xlnk, batchsize=self.batchsize)

        self.convAll = ConvLayer('conv_all', 192, 6, self.xlnk, runFactor=2, batchsize = self.batchsize)

        softLayer = ['fc_full', 'fc_detection1', 'fc_detection2', 'fc_gender1', 'fc_gender2','fc_landmarks1',
                     'fc_landmarks2', 'fc_visibility1', 'fc_visibility2', 'fc_pose1', 'fc_pose2']

        self.weights = {}
        for sl in softLayer:
            w = np.load(os.path.dirname(os.path.realpath(__file__)) + "/weights/" + sl + "/W.npy")
            b = np.load(os.path.dirname(os.path.realpath(__file__)) + "/weights/" + sl + "/b.npy")
            self.weights[sl] = (w, b)

        memStat = self.xlnk.cma_stats()
        logger.info("CMA Stat : " + str(memStat['CMA Memory Usage']) + ' / ' + str(memStat['CMA Memory Available']+memStat['CMA Memory Usage']) +
         " [ " + str( int((memStat['CMA Memory Usage']/ (memStat['CMA Memory Available']+memStat['CMA Memory Usage']) )*100)) + " % ] ") 


    def __call__(self, img):
        self.batchsize = len(img)
        logger.info('Start computation on FPGA with batch size ' + str(self.batchsize))

        if self.batchsize > self.batchsizeMax:
            raise Exception('Batch size exceed the max threshold')

        b = 0
        while b < self.batchsize:
            np.copyto(self.cmaIn[b], img[b].ravel())
            b += 1

        outMax1 = self.max1(self.conv1(self.cmaIn))

        outConv1a = self.conv1a(outMax1)

        outConv3 = self.conv3(self.max2(self.conv2(outMax1)))
        outConv3a = self.conv3a(outConv3)

        outMax5 = self.max5(self.conv5(self.conv4(outConv3)))

        self.concat(outConv1a, outConv3a, outMax5)

        outConvAll = self.convAll(self.inConvAll)

        res = self.softLayer(outConvAll)

        logger.info('End computation')
        return res


    def concat(self, first, second, third):
        b = 0
        while b < self.batchsize:
            f = first[b].reshape((36, 256))
            f = f.transpose()
            f = f.reshape(36 * 256)

            s = second[b].reshape((36, 256))
            s = s.transpose()
            s = s.reshape(36 * 256)

            t = third[b].reshape((36, 256))
            t = t.transpose()
            t = t.reshape(36 * 256)

            full = np.concatenate((f, s, t))

            full = full.reshape(768, 36).transpose().flatten()

            np.copyto(self.inConvAll[b], full)

            b += 1


    def softLayer(self, input):
        res = []
        b = 0
        while b < self.batchsize:
            inFcFull = np.transpose(input[b].reshape(6, 6, 192), (2, 0, 1)).flatten()

            outFcFull = self.fcLayer(inFcFull, self.weights['fc_full'])
            outFcFull = outFcFull * (outFcFull > 0)

            outFcDet2 = self.featDetect(outFcFull, self.weights['fc_detection1'], self.weights['fc_detection2'])
            (face_det, face_val) = self.softMax(outFcDet2)
            
            if face_det > 0.25:
                logger.info('Possible region found on batch index ' + str(b))

            outFcGen2 = self.featDetect(outFcFull, self.weights['fc_gender1'], self.weights['fc_gender2'])
            (gen_det, gen_val) = self.softMax(outFcGen2)

            outFcLan2 = self.featDetect(outFcFull, self.weights['fc_landmarks1'], self.weights['fc_landmarks2'])

            outFcVis2 = self.featDetect(outFcFull, self.weights['fc_visibility1'], self.weights['fc_visibility2'])

            outFcPos2 = self.featDetect(outFcFull, self.weights['fc_pose1'], self.weights['fc_pose2'])

            res.append({'img': self.cmaIn[b], 'detection': face_val,
                    'landmark': outFcLan2, 'visibility': outFcVis2,
                    'pose': outFcPos2, 'gender': gen_val})

            b += 1

        return res


    def fcLayer(self, x, y):
        (w, b) = y
        out = np.matmul(w, x)
        out = np.sum([b, out], axis=0)
        return out


    def featDetect(self, x, y1, y2):
        out = self.fcLayer(x, y1)
        out = out * (out > 0)
        out = self.fcLayer(out, y2)
        return out


    def softMax(self, x):
        e_x = np.exp(x)
        vals = e_x / e_x.sum()
        val = vals[1]
        c = np.argmax(vals)
        
        return (c, val)


def print_dma_status():
    print("====  From memory to FIFO  ====")
    print("MM to Stream Control: 0x" +
          format(dma.read(0x0), '0x'))
    print("             Binary : 0b" +
          format(dma.read(0x0), '0b'))
    print("MM to Stream Status : 0x" +
          format(dma.read(0x4), '0x'))
    print("             Binary : 0b" +
          format(dma.read(0x4), '0b'))

    print("\n==== From FIFO to Memory ====")
    print("Stream to MM Control: 0x" +
          format(dma.read(0x30), '0x'))
    print("             Binary : 0b" +
          format(dma.read(0x30), '0b'))
    print("Stream to MM Status : 0x" +
          format(dma.read(0x34), '0x'))
    print("             Binary : 0b" +
          format(dma.read(0x34), '0b'))


