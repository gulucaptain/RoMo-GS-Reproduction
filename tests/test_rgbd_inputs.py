import torch
from scripts.rgbd_experiment import initial_colors


def inputs():
    return {'rgb':torch.rand(3,1,2,3),'blur':torch.tensor([[[[1.,0.,0.],[0.,0.,1.]]],
                                                       [[[0.,1.,0.],[0.,0.,0.]]],
                                                       [[[1.,1.,1.],[1.,1.,1.]]]]),
            'native_mask':torch.ones(3,1,2,dtype=torch.bool),'controlled_mask':torch.tensor([[[True,False]],[[True,False]],[[True,True]]]),
            'init_means':torch.zeros(2,3),'fusion_groups':torch.tensor([0,1,0]),
            'fusion_images':torch.tensor([0,0,1]),'fusion_pixels':torch.tensor([0,1,0])}


def test_controlled_colors_do_not_leak_clean_or_test_rgb():
    data=inputs();a=initial_colors(data,'controlled')
    data['rgb'].fill_(.973)  # All clean RGB changes, including training/test.
    data['blur'][2].fill_(.287)  # Held-out observation changes as well.
    b=initial_colors(data,'controlled')
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(a[0],torch.tensor([.5,.5,.001]))


def test_no_valid_controlled_color_uses_neutral_initialization():
    data=inputs();data['controlled_mask'][:2]=False
    torch.testing.assert_close(initial_colors(data,'controlled'),torch.full((2,3),.5))
