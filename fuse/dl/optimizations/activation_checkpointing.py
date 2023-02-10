from torch import nn
#from torch.utils.checkpoint import checkpoint
import torch



# class ActivationCheckpointer:
#     """
#     Assumes that the model forward function returns a dictionary which contains 'losses' which points to a list of 
#     torch tensors that we can do backward on
#     """
#     def __init__(self):
#         pass

#     def forward(self, func, **func_kwargs):






def custom_forward_and_backward(config, model, x, targets, regularization_terms,
                                        do_backward=False,
                                        use_cpu_ram_paging=False,
                                        calc_loss=True):   
    if do_backward and not calc_loss:
        raise Exception('calculate_loss cannot be disabled when do_backward is enabled.')

    sample_extractor = SampleExtractor([x], move_to_gpu=True)

    # compute_vol_features = make_seq_func(
    #    sample_extractor,
    #    #lambda x: x.permute(1, 0, 2, 3),
    #    (model, {'run': 'extract_features'}),
    #    GlobalMaxPooling()
    # )

    use_feature_extractor_checkpointing = config['use_feature_extractor_checkpointing']
    assert use_feature_extractor_checkpointing is not None

    # build a list of sequences. A seuqnece is simply a python function

    funcs_seq = []
    #import ipdb;ipdb.set_trace()
    for idx in range(len(use_feature_extractor_checkpointing)):

        start = use_feature_extractor_checkpointing[idx][0]
        end = use_feature_extractor_checkpointing[idx][1]

        model_func_tuple = (model, {
            'curr_part': (start, end),
            'run': 'extract_features'
        })

        if idx == 0:
            #for the first one, add the sample extractor
            funcs_seq += [make_seq_func(
                sample_extractor,
                model_func_tuple
            )]
        else:
            funcs_seq += [make_seq_func(
                model_func_tuple)
            ]
    
    ### add the feature reduction phase to the last function sequence
    ### (note: assumes that model_func_tuple is unaltered after the loop)

    funcs_seq[-1] = make_seq_func(
        model_func_tuple,
        # GlobalMaxPooling(),
        (model, {'run': 'reduce_features'}),
    )

    funcs_seq += [
        make_seq_func(
            (model, {'run': 'classify'})
        )
    ]

    def _calculate_loss(raw_predictions):
        total_loss, individual_losses, regularization_loss = model.get_total_loss(raw_predictions, targets,
                                                                                  **regularization_terms)
        return total_loss, individual_losses, regularization_loss

    def execute_func(f, x):
        if isinstance(x, list) or isinstance(x, tuple):
            # x = f(*x)
            x = checkpoint(f, *x)
        else:
            # x = f(x)
            x = checkpoint(f, x)
        return x

    
    if do_backward:
        
        if False:
            cprint('remove this - overriding checkpointing for debugging!!!!!\n' * 4, 'red')
            assert False

            x = 0

            for f_idx in range(len(funcs_seq)):
                x = execute_func(funcs_seq[f_idx], x)

            total_loss, individual_losses, regularization_loss = _calculate_loss(x)

            raw_predictions = x

            total_loss.backward()
        else:
            raw_predictions, total_loss, individual_losses, regularization_loss = calculate_grads_simulating_flat_checkpointing(
                funcs_seq,
                # stored_features[slices_chunk_step_num].grad,
                0,
                loss_calc_func=_calculate_loss,
                cpu_ram_paging=use_cpu_ram_paging
            )
    else:
        with torch.set_grad_enabled(False):
            entire_forward_sequence = make_seq_func(*funcs_seq)
            raw_predictions = entire_forward_sequence(0)
            if calc_loss:
                total_loss, individual_losses, regularization_loss = _calculate_loss(raw_predictions)
            else:
                total_loss, individual_losses, regularization_loss = None, None, None

    return raw_predictions, total_loss, individual_losses, regularization_loss




class SampleExtractor(nn.Module):
    '''
    Used as a trick to make sure that checkpointing doesn't need to store in GPU memory
    '''

    def __init__(self, mb, move_to_gpu=True, verbose=0):
        super().__init__()
        self.move_to_gpu = move_to_gpu
        self.mb = mb
        self.verbose = verbose
        if self.verbose > 0:
            print('SampleExtractor initialized with mb = ')
            for x in self.mb:
                print('shape=', x.shape)

    def forward(self, sample_idx, slices_range=None):
        if slices_range is not None:
            assert 2 == slices_range.shape[0]

        if not isinstance(sample_idx, int):
            #import ipdb;ipdb.set_trace()
            sample_idx = int(sample_idx.item())

        if self.verbose > 0:
            print('SampleExtractor:: ', sample_idx, 'slices range:', slices_range)

        if slices_range is not None:
            s = self.mb[sample_idx][0:1, slices_range[0].item():slices_range[1].item(),
                ...].clone()  # should I also do .copy() ?
        else:
            s = self.mb[sample_idx]  # .clone() ??

        if self.verbose > 0:
            print('extracted size', s.shape)
        if self.move_to_gpu:
            s = s.cuda()

        return s

def make_seq_func(*funcs, verbose=0):
    def foo(*args):
        if verbose > 0:
            print('running sequence of : ', end='')
            for f in funcs:
                print('{}, '.format(type(f)), end='')
            print('')
        # gc.collect()
        # torch.cuda.empty_cache()
        for idx, func in enumerate(funcs):
            # gc.collect()
            if isinstance(func, tuple):
                func, kwargs = func[0], func[1]
            else:
                kwargs = {}
            func = partial(func, **kwargs)
            if idx == 0:
                x = func(*args)
            else:
                if isinstance(x, tuple):
                    x = func(*x)
                else:
                    x = func(x)
            # gc.collect()
            # torch.cuda.empty_cache()
        return x

    return foo



def calculate_grads_simulating_flat_checkpointing(funcs, *inputs, output_grad=None, loss_calc_func=None,
                                                  cpu_ram_paging=False):
    if inputs is None:
        raise Exception('you must provide inputs tensor(s)')

    if not ((output_grad is not None) or (loss_calc_func is not None)):
        raise Exception('at least one of output_grad or loss_calc_func must be provided!')

    if (output_grad is not None) and (loss_calc_func is not None):
        raise Exception('only one of output_grad or loss_calc_func may be used!')
    # simulate standard checkpointing, only doing forward twice per func

    # assert output_grad is not None

    saved_inputs = {}

    def execute_func(f, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = f(*x)
        else:
            x = f(x)
        return x

    with torch.no_grad():
        x = inputs
        saved_inputs[0] = x
        if cpu_ram_paging:
            # saved_inputs[0] = saved_inputs[0].cpu()
            saved_inputs[0] = move_tensors_to_device(saved_inputs[0], torch.device('cpu'))
        # for f in funcs[:-1]:

        forward_only_steps = len(funcs) - 1

        for f_idx in range(forward_only_steps):
            x = execute_func(funcs[f_idx], x)
            saved_inputs[f_idx + 1] = detach_tensors(x)
            if cpu_ram_paging:
                # saved_inputs[f_idx + 1] = saved_inputs[f_idx + 1].cpu()
                saved_inputs[f_idx + 1] = move_tensors_to_device(saved_inputs[f_idx + 1], torch.device('cpu'))

    del inputs

    '''
    if loss_calc_func is not None:
        assert callable(loss_calc_func)
        with torch.set_grad_enabled(True):
            #x.requires_grad_()
            x.requires_grad_()
            raw_predictions = execute_func(funcs[-1], x)
            loss_calc_res = loss_calc_func(raw_predictions)
    '''

    saved_grad = output_grad

    loss_calc_res = None

    del x

    def _print_info(t, name=''):
        print(name, ': shape=', t.shape, ' is_contiguous=', t.is_contiguous())

    for f_idx in range(len(funcs) - 1, -1, -1):
        ####print('checkpoint part ', f_idx)
        # forward and then backward of each part separately
        # trying to delete everything that isn't needed anymore

        curr_input = saved_inputs[f_idx]
        if cpu_ram_paging:
            # curr_input = curr_input.cuda()
            curr_input = move_tensors_to_device(curr_input, torch.device('cuda'))

        ####print('converting inputs to contiguous...')
        curr_input = to_contiguous(curr_input)
        ####print('done converting inputs')

        # curr_input.requires_grad_()
        set_requires_grad(curr_input)

        if isinstance(curr_input, list) or isinstance(curr_input, tuple):
            curr_output = funcs[f_idx](*curr_input)
        else:
            curr_output = funcs[f_idx](curr_input)

        del saved_inputs[f_idx]

        # gc.collect()

        # if loss_func is not None and f_idx == len(funcs) - 1:
        #    curr_output = loss_func(curr_output)

        if (loss_calc_func is not None) and (f_idx == len(funcs) - 1):

            raw_predictions = curr_output  # I should probably detach the raw_predictions (after the backward...)
            loss_calc_res = loss_calc_func(raw_predictions)
            loss_calc_res[0].backward()  # the first loss_calc_res element is the total_loss

            # now, detach it so it won't occupy memory
            loss_calc_res = detach_tensors(loss_calc_res)
        else:
            # added to try and solve the nvidia error:
            # RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
            ##_print_info(curr_output,'bef curr_output')
            ##_print_info(saved_grad, 'bef saved_grad')
            saved_grad = saved_grad.contiguous()  # https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930
            curr_output = curr_output.contiguous()  # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
            ##_print_info(curr_output, 'aft curr_output')
            ##_print_info(saved_grad, 'aft saved_grad')
            # https://discuss.pytorch.org/t/resolved-batchnorm1d-cudnn-status-not-supported/3049
            if False:
                all_mem_tensors = list(get_in_mem_torch_tensors())
                tensors_count = len(all_mem_tensors)
                print(f'found in memory {tensors_count} gpu tensors')
                for t in all_mem_tensors:
                    if not t[1].is_contiguous():
                        print('found non contiguous tensor:')
                        print(t[0], str(t[1].shape))
                del all_mem_tensors
                # print('allocated=' + str(torch.cuda.memory_allocated(0) / (10 ** 9)) + ' GB    cached=' + str(torch.cuda.memory_cached(0) / (10 ** 9)) + ' GB')
            curr_output.backward(saved_grad)
            # https://discuss.pytorch.org/t/contiguous-and-permute/20673/2
        if 0 == f_idx:
            if loss_calc_func is not None:
                return tuple([raw_predictions] + list(loss_calc_res))
            else:
                return

        if torch.is_tensor(curr_input) and curr_input.grad is not None:
            saved_grad = detach_tensors(curr_input.grad)  # .detach()
        else:
            saved_grad = None

        del curr_input
        del curr_output

        # gc.collect()


def move_tensors_to_device(inputs, dev):
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        moved_inputs = [x.to(dev) if torch.is_tensor(x) else x for x in inputs]
        return moved_inputs
    if isinstance(inputs, dict):  # for now, return dicts as is
        return inputs
    return inputs.to(dev)


def to_contiguous(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        # for t in tensors:
        #    if is_float_tens(t):
        #        t.requires_grad_()
        converted = [t.contiguous() if is_float_tens(t) else t for t in tensors]
        # _ = [print(t.shape) for t in converted]
        for t in converted:
            if is_float_tens(t):
                print(t.shape)
        return type(tensors)(converted)
    else:
        if is_float_tens(tensors):
            return tensors.contiguous()
        else:
            return tensors

def is_float_tens(x):
    if not torch.is_tensor(x):
        return False
    return x.dtype in [torch.float16, torch.float32, torch.float64]                    


def detach_tensors(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        moved_inputs = [x.detach() if torch.is_tensor(x) else x for x in tensors]
        return moved_inputs
    if isinstance(tensors, dict):  # for now, return dicts as is
        return tensors
    return tensors.detach()




def set_requires_grad(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        for t in tensors:
            if is_float_tens(t):
                t.requires_grad_()
    else:
        if is_float_tens(tensors):
            tensors.requires_grad_()



if __name__ == '__main__' :
    import torch
    from torch import nn
    from torchvision import models
    from typing import Optional, Any
    from tqdm import trange
    import tempfile
    import os
    from fuse.utils.file_io import save_hdf5_safe, load_hdf5
    import uuid
    

    class MyModel(nn.Module):
        def __init__(self, *, layers_num:int, filters_num:int, shared_params:bool):
            super(MyModel, self).__init__()
            print('layers_num=',layers_num)
            print('filters_num=',filters_num)
            print('shared_params=',shared_params)
            self.initial_conv = nn.Conv2d(3, filters_num, 1)
            
            if shared_params:
                self.conv = nn.Conv2d(filters_num, filters_num, 1)
            else:
                self.convs = nn.ModuleList([
                    nn.Conv2d(filters_num, filters_num, 3) for _ in range(layers_num)
                ])

            self.shared_params = shared_params
            self.layers_num = layers_num

        
        def forward(self, x):
            x = self.initial_conv(x)
            for step in range(self.layers_num):
                if self.shared_params:
                    conv = self.conv
                else:
                    conv = self.convs[step]
                x = conv(x)
                #print(x.shape)
                #print('step=',step)
            
            return x

    
    #if True:

    model = MyModel(layers_num=60, filters_num=4096, shared_params=True).cuda()
    model.train()
    torch.manual_seed(1337)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)


    #### offload to CPU

    def pack_hook_to_cpu(tensor: torch.Tensor):
        address = tensor.data_ptr()
        #print(f'pack_hook_to_cpu ({address})')
        return dict(
            orig_device = tensor.device,
            cpu_tensor = tensor.to('cpu'),
            address = address,
        )

    def unpack_hook_from_cpu(packed:Any):
        #print(f'unpack_hook_from_cpu ({packed["address"]})')
        tens = packed['cpu_tensor'].to(packed['orig_device'])
        return tens

    #### offload to disk
    #see https://pytorch.org/docs/stable/notes/autograd.html#hooks-for-saved-tensors
    class SelfDeletingTempFile():
        def __init__(self):
            #tmp_dir = os.path.join(tempfile.gettempdir(), 'checkpointed_activations')

            #tmp_dir = os.path.join(os.path.realpath('~/'), 'temp/checkpointed_activations')
            tmp_dir = '/dccstor/fmm/users/yoels/temp/checkpointed_activations'

            os.makedirs(tmp_dir, exist_ok=True)
            self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

        def __del__(self):
            os.remove(self.name)

    def pack_hook_to_disk(tensor):
        temp_file = SelfDeletingTempFile()
        #torch.save(tensor, temp_file.name)
        save_hdf5_safe(temp_file.name, use_blosc=False, tensor=tensor.cpu().numpy())
        return temp_file

    def unpack_hook_from_disk(temp_file):
        loaded = load_hdf5(temp_file.name)
        tensor = loaded['tensor']
        tensor = torch.from_numpy(tensor).cuda()
        return tensor
        
        #return torch.load(temp_file.name)


    for i in trange(100_000):
        x = torch.randn(1, 3, 224, 224).cuda()
        print('forward:')
        with torch.autograd.graph.saved_tensors_hooks(pack_hook_to_disk, unpack_hook_from_disk):
        #with torch.autograd.graph.saved_tensors_hooks(pack_hook_to_cpu, unpack_hook_from_cpu):
        #with torch.autograd.graph.save_on_cpu(pin_memory=True):
        #with torch.autograd.graph.save_on_cpu(pin_memory=False):
        #if True:
            x = model(x)
        print('backward:')
        x.mean().backward()
        
        #print(model.convs[0].weight.grad.max())
        opt.step()
        opt.zero_grad()





