from networks import *
from training import train, train_mirroring

net = BackpropagationNet(num_hidden=512)
params = {'lr': 0.2}

# net = FinalLayerUpdateNet()
# params = {'lr': 0.2}

# net = FeedbackAlignmentNet()
# params = {'lr': 0.2}

# net = SignSymmetryNet()
# params = {'lr': 0.2}

# net = WeightMirroringNet()
# params = {'lr_forward': 0.1, 'lr_backward': 0.005, 'weight_decay_backward': 0.2}
# train_mirroring(net, params)

# net = TargetPropagationNet()
# params = {'lr_final': 0.5, 'lr_forward': 0.3, 'lr_backward': 0.001}

# net = EquilibriumPropagationNet(num_hidden=30)
# params = {'lr1': 0.1, 'lr2': 0.05, 'step_size': 0.5, 'beta': 1, 'steps_free': 20, 'steps_clamped': 4}

train(net, params)


