import torch
pretrained_weights  = torch.load('fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20190516-42e6f62d.pth')

num_class = 2

# store = []
# for index, name in enumerate(pretrained_weights['state_dict']):
#     store.append(name)
# a = store[500:]

# b = pretrained_weights['state_dict']['bbox_head.fcos_cls.weight']
# c = pretrained_weights['state_dict']['bbox_head.fcos_cls.bias']
# d = pretrained_weights['state_dict']['bbox_head.scales.0.scale']
# e = pretrained_weights['state_dict']['bbox_head.scales.1.scale']

pretrained_weights['state_dict']['bbox_head.fcos_cls.weight'].resize_(num_class, 1024, 3, 3)
pretrained_weights['state_dict']['bbox_head.fcos_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.fcos_reg.weight'].resize_(num_class*4, 1024, 3, 3)
pretrained_weights['state_dict']['bbox_head.fcos_reg.bias'].resize_(num_class*4)

torch.save(pretrained_weights, "fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_%d.pth"%num_class)
