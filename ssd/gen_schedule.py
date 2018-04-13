from collections import namedtuple
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

def output_height(wkl):
    return (wkl.height + wkl.hpad * 2 - wkl.hkernel) // wkl.hstride + 1

def output_width(wkl):
    return (wkl.width + wkl.wpad * 2 - wkl.wkernel) // wkl.wstride + 1

def gen(wkl):
    return 'ic = %d, oc = %d, ow = %d' % (wkl.in_filter, wkl.out_filter, output_width(wkl))

workloads = [
    # SSD Resnet50_v2 0-22
    Workload('float32', 'float32', 512, 512, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('float32', 'float32', 128, 128, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 128, 128, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 128, 128, 64, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 128, 128, 256, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 128, 128, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 128, 128, 128, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 64, 64, 128, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 128, 128, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 64, 64, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 64, 64, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 64, 64, 256, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 32, 32, 256, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 1024, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 32, 32, 1024, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 32, 32, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 16, 16, 512, 2048, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 2048, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 16, 16, 2048, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 16, 16, 512, 512, 3, 3, 1, 1, 1, 1),
    # SSD Resnet50_v2 others 23-42
    # layer2
    Workload('float32', 'float32', 16, 16, 2048, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 16, 16, 256, 512, 3, 3, 1, 1, 2, 2),
    # layer3
    Workload('float32', 'float32', 8, 8, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 8, 8, 128, 256, 3, 3, 1, 1, 2, 2),
    # layer4
    Workload('float32', 'float32', 4, 4, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 4, 4, 128, 256, 3, 3, 1, 1, 2, 2),
    # layer5
    Workload('float32', 'float32', 2, 2, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 2, 2, 128, 128, 3, 3, 1, 1, 2, 2),
    # loc_preds
    Workload('float32', 'float32', 32, 32, 1024, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 16, 16, 2048, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 8, 8, 512, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 4, 4, 256, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 2, 2, 256, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 1, 1, 128, 16, 3, 3, 1, 1, 1, 1),
    # cls_preds
    Workload('float32', 'float32', 32, 32, 1024, 84, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 16, 16, 2048, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 8, 8, 512, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 4, 4, 256, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 2, 2, 256, 84, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 1, 1, 128, 84, 3, 3, 1, 1, 1, 1),
]

for i in range(len(workloads)):
    print(str(i) + ' ' + gen(workloads[i]))
