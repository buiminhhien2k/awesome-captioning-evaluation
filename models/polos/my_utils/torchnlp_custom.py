import typing
import torch
import urllib.request
import zipfile
import tarfile
import os

class SequenceBatch(typing.NamedTuple):
    tensor: torch.Tensor
    lengths: torch.Tensor

DEFAULT_PADDING_INDEX = 0
def pad_tensor(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)

def stack_and_pad_tensors(batch, padding_index=DEFAULT_PADDING_INDEX, dim=0):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        SequenceBatch: Padded tensors and original lengths of tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)

    return SequenceBatch(padded, lengths)

class Encoder(object):
    """
    Base class for a encoder employing an identity function.

    Args:
        enforce_reversible (bool, optional): Check for reversibility on ``Encoder.encode`` and
          ``Encoder.decode``. Formally, reversible means:
          ``Encoder.decode(Encoder.encode(object_)) == object_``.
    """

    def __init__(self, enforce_reversible=False):
        self.enforce_reversible = enforce_reversible

    def encode(self, object_):
        """ Encodes an object.

        Args:
            object_ (object): Object to encode.

        Returns:
            object: Encoding of the object.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            encoded_decoded = self.decode(self.encode(object_))
            self.enforce_reversible = True
            if encoded_decoded != object_:
                raise ValueError('Encoding is not reversible for "%s"' % object_)

        return object_


    def batch_encode(self, iterator, *args, **kwargs):
        """
        Args:
            batch (list): Batch of objects to encode.
            *args: Arguments passed to ``encode``.
            **kwargs: Keyword arguments passed to ``encode``.

        Returns:
            list: Batch of encoded objects.
        """
        return [self.encode(object_, *args, **kwargs) for object_ in iterator]


    def decode(self, encoded):
        """ Decodes an object.

        Args:
            object_ (object): Encoded object.

        Returns:
            object: Object decoded.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            decoded_encoded = self.encode(self.decode(encoded))
            self.enforce_reversible = True
            if decoded_encoded != encoded:
                raise ValueError('Decoding is not reversible for "%s"' % encoded)

        return encoded


    def batch_decode(self, iterator, *args, **kwargs):
        """
        Args:
            iterator (list): Batch of encoded objects.
            *args: Arguments passed to ``decode``.
            **kwargs: Keyword arguments passed to ``decode``.

        Returns:
            list: Batch of decoded objects.
        """
        return [self.decode(encoded, *args, **kwargs) for encoded in iterator]


class TextEncoder(Encoder):

    def decode(self, encoded):
        """ Decodes an object.

        Args:
            object_ (object): Encoded object.

        Returns:
            object: Object decoded.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            decoded_encoded = self.encode(self.decode(encoded))
            self.enforce_reversible = True
            if not torch.equal(decoded_encoded, encoded):
                raise ValueError('Decoding is not reversible for "%s"' % encoded)

        return encoded


    def batch_encode(self, iterator, *args, dim=0, **kwargs):
        """
        Args:
            iterator (iterator): Batch of text to encode.
            *args: Arguments passed onto ``Encoder.__init__``.
            dim (int, optional): Dimension along which to concatenate tensors.
            **kwargs: Keyword arguments passed onto ``Encoder.__init__``.

        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        return stack_and_pad_tensors(
            super().batch_encode(iterator), padding_index=self.padding_index, dim=dim)


    def batch_decode(self, tensor, lengths, dim=0, *args, **kwargs):
        """
        Args:
            batch (list of :class:`torch.Tensor`): Batch of encoded sequences.
            lengths (torch.Tensor): Original lengths of sequences.
            dim (int, optional): Dimension along which to split tensors.
            *args: Arguments passed to ``decode``.
            **kwargs: Key word arguments passed to ``decode``.

        Returns:
            list: Batch of decoded sequences.
        """
        return super().batch_decode(
            [t.squeeze(0)[:l] for t, l in zip(tensor.split(1, dim=dim), lengths)])

def lengths_to_mask(*lengths, **kwargs):
    """ Given a list of lengths, create a batch mask.

    Example:
        >>> lengths_to_mask([1, 2, 3])
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[ True, False],
                 [False, False]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]]])

    Args:
        *lengths (list of int or torch.Tensor)
        **kwargs: Keyword arguments passed to ``torch.zeros`` upon initially creating the returned
          tensor.

    Returns:
        torch.BoolTensor
    """
    # Squeeze to deal with random additional dimensions
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][tuple(slice(int(l)) for l in length)].fill_(1)
    return mask.bool()


def download_file_maybe_extract(url, directory, filename=None, extract=True):
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        filename = os.path.basename(url)

    file_path = os.path.join(directory, filename)

    # Download if not exists
    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Saved to {file_path}")
    else:
        print(f"File already exists: {file_path}")

    # Extract if needed
    if extract:
        if zipfile.is_zipfile(file_path):
            print("Extracting zip file...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(directory)

        elif tarfile.is_tarfile(file_path):
            print("Extracting tar file...")
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(directory)

        else:
            print("No extraction needed.")

    return file_path

def is_namedtuple(object_):
    return hasattr(object_, '_asdict') and isinstance(object_, tuple)


def collate_tensors(batch, stack_tensors=torch.stack):
    """ Collate a list of type ``k`` (dict, namedtuple, list, etc.) with tensors.

    Inspired by:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L31

    Args:
        batch (list of k): List of rows of type ``k``.
        stack_tensors (callable): Function to stack tensors into a batch.

    Returns:
        k: Collated batch of type ``k``.

    Example use case:
        This is useful with ``torch.utils.data.dataloader.DataLoader`` which requires a collate
        function. Typically, when collating sequences you'd set
        ``collate_fn=partial(collate_tensors, stack_tensors=encoders.text.stack_and_pad_tensors)``.

    Example:

        >>> import torch
        >>> batch = [
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ... ]
        >>> collated = collate_tensors(batch)
        >>> {k: t.size() for (k, t) in collated.items()}
        {'column_a': torch.Size([2, 5]), 'column_b': torch.Size([2, 5])}
    """
    if all([torch.is_tensor(b) for b in batch]):
        return stack_tensors(batch)
    if (all([isinstance(b, dict) for b in batch]) and
            all([b.keys() == batch[0].keys() for b in batch])):
        return {key: collate_tensors([d[key] for d in batch], stack_tensors) for key in batch[0]}
    elif all([is_namedtuple(b) for b in batch]):  # Handle ``namedtuple``
        return batch[0].__class__(**collate_tensors([b._asdict() for b in batch], stack_tensors))
    elif all([isinstance(b, list) for b in batch]):
        # Handle list of lists such each list has some column to be batched, similar to:
        # [['a', 'b'], ['a', 'b']] → [['a', 'a'], ['b', 'b']]
        transposed = zip(*batch)
        return [collate_tensors(samples, stack_tensors) for samples in transposed]
    else:
        return batch