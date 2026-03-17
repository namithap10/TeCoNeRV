import math
import os
import tempfile

import torch

import torchac
from dahuffman import HuffmanCodec


def arithmetic_encoding(x):
    """ Arithmetic encoding for a tensor with torchac. """    
    with torch.no_grad():
        x = x.detach().view(-1).cpu()
        sym, inverse, counts = x.unique(return_inverse=True, return_counts=True)
        inverse = inverse.to(torch.int16)
        counts = torch.concat([torch.zeros([1], dtype=torch.int64, device=counts.device), counts])
        cdf = torch.cumsum(counts, dim=0).float() / counts.sum().float()
        byte_stream = torchac.encode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), inverse, check_input_bounds=True, needs_normalization=True)        
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), byte_stream).int()
        assert inverse_out.equal(inverse)
        x_out = sym[inverse_out.long()]
        assert x_out.equal(x)
        return sym, cdf, byte_stream


def arithmetic_decoding(byte_stream, sym, cdf, shape):
    """ Arithmetic decoding for a tensor with torchac. """    
    with torch.no_grad():
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(shape), 1), byte_stream).int()
        x_out = sym[inverse_out.long()].view(shape)
        return x_out


def compress_tensor(x, dim, output_dir, name):
    """ Compress a tensor and store its meta data. """
    # assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    if x.numel() > 0:
        # arithmetic_encoding requires postive integers
        offset = x.min() if x.numel() > 0 else 0
        x -= offset

        if dim is None:
            sym, cdf, bits = arithmetic_encoding(x)
            bits_length = len(bits)
        else:
            sym = []
            cdf = []
            bits = None
            bits_length = []
            x_splits = torch.split(x, 1, dim=dim)
            for x_i in x_splits:
                sym_i, cdf_i, bits_i = arithmetic_encoding(x_i.contiguous())
                sym.append(sym_i)
                cdf.append(cdf_i)
                bits = bits + bits_i if bits is not None else bits_i
                bits_length.append(len(bits_i))

        meta = {
            'dim': dim,
            'sym': sym,
            'cdf': cdf,
            'bits_length': bits_length,
            'shape': x.shape,
            'offset': offset
        }
    else:
        meta = {
            'dim': None,
            'sym': None,
            'cdf': None,
            'bits_length': 0,
            'shape': None,
            'offset': None
        }

    # Either compress the tensor and return the meta data, or return the tensor
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # byte_stream
        if x.numel() > 0:
            with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
                f.write(bits)

        return meta
    else:
        return meta, bits


def decompress_tensor(state_dict, output_dir, name, bits):
    """ Decompress a tensor. """
    # Read meta file
    dim = state_dict['dim']
    sym = state_dict['sym']
    cdf = state_dict['cdf']
    bits_length = state_dict['bits_length']
    shape = state_dict['shape']
    offset = state_dict['offset']

    # Read bitstream
    if (isinstance(bits_length, (int, float)) and bits_length > 0) or (isinstance(bits_length, list) and len(bits_length) > 0):
        if output_dir is not None:
            with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
                byte_stream = f.read()
        else:
            byte_stream = bits

        if dim is None:
            x = arithmetic_decoding(byte_stream, sym, cdf, shape)
        else:
            x_splits = []
            bits_count = 0
            for i in range(shape[dim]):
                x_splits.append(arithmetic_decoding(byte_stream[bits_count:bits_count + bits_length[i]], sym[i], cdf[i], list(shape[:dim]) + [1] + list(shape[dim+1:])))
                bits_count += bits_length[i]
            x = torch.concat(x_splits, dim=dim)

        x += offset.to(x.device)
    else:
        x = torch.zeros([0])

    return x


def _get_target_keys(model):
    target_keys = []
    for k in model.state_dict().keys():
        if k.startswith(model.bitstream_prefix) and not 'mask' in k:
            target_keys.append(k)
    return target_keys


def _get_mask_key(state_dict, k):
    if k.endswith('.weight.original'):
        for i in range(5): # assuming upto 5 parameterizations
            mask_k = k.replace('.weight.original', f'.weight.{i}.mask')
            if mask_k in state_dict:
                return mask_k
    return None


def compress_tensor_huffman(tensor):
    """Compress tensor using Huffman coding"""
    values = tensor.flatten().tolist()
    values = [int(x) for x in values]
    
    codec = HuffmanCodec.from_data(values)
    encoded = codec.encode(values)
    
    return {
        'encoded': encoded,
        'codec': codec,
        'bits_length': len(encoded) * 8  # Length in bits
    }

