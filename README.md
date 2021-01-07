# inception_v3_dp
I manually replaced batch_norm with group_norm in inception_v3, but there is still a problem in cifar10_inception3_dp.py: `AttributeError: 'Parameter' object has no attribute 'grad_sample'`
