import binascii
import os
import uuid as _uu
from math import ceil, log


def int_to_string(number: int, alphabet: list[str], padding: int | None = None) -> str:
    """
    Convert a number to a string, using the given alphabet.

    The output has the most significant digit first.
    """
    output = ""
    alpha_len = len(alphabet)
    while number:
        number, digit = divmod(number, alpha_len)
        output += alphabet[digit]
    if padding:
        remainder = max(padding - len(output), 0)
        output = output + alphabet[0] * remainder
    return output[::-1]


def string_to_int(string: str, alphabet: list[str]) -> int:
    """
    Convert a string to a number, using the given alphabet.

    The input is assumed to have the most significant digit first.
    """
    number = 0
    alpha_len = len(alphabet)
    for char in string:
        number = number * alpha_len + alphabet.index(char)
    return number


class ShortUUID:
    """
    Instantiate a ShortUUID object.

    >>> shortuuid = ShortUUID()
    You can then generate a short UUID:

    >>> shortuuid.uuid()
    'vytxeTZskVKR7C7WgdSP3d'
    If you prefer a version 5 UUID, you can pass a name (DNS or URL) to the call and it will be used as a namespace (uuid.NAMESPACE_DNS or uuid.NAMESPACE_URL) for the resulting UUID:

    >>> shortuuid.uuid(name="example.com")
    'exu3DTbj2ncsn9tLdLWspw'

    >>> shortuuid.uuid(name="<http://example.com>")
    'shortuuid.uuid(name="<http://example.com>")'
    You can also generate a cryptographically secure random string (using os.urandom() internally) with:

    >>> shortuuid.ShortUUID().random(length=22)
    'RaF56o2r58hTKT7AYS9doj'
    To see the alphabet that is being used to generate new UUIDs:

    >>> shortuuid.get_alphabet()
    '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    If you want to use your own alphabet to generate UUIDs, use set_alphabet():

    >>> shortuuid.set_alphabet("aaaaabcdefgh1230123")
    >>> shortuuid.uuid()
    '0agee20aa1hehebcagddhedddc0d2chhab3b'
    The default alphabet matches the regex [2-9A-HJ-NP-Za-km-z]{22}.

    shortuuid will automatically sort and remove duplicates from your alphabet to ensure consistency:

    >>> shortuuid.get_alphabet()
    '0123abcdefgh'
    If the default 22 digits are too long for you, you can get shorter IDs by just truncating the string to the desired length. The IDs won't be universally unique any longer, but the probability of a collision will still be very low.

    To serialize existing UUIDs, use encode() and decode():

    >>> import uuid
    >>> u = uuid.uuid4()
    >>> u
    UUID('6ca4f0f8-2508-4bac-b8f1-5d1e3da2247a')

    >>> s = shortuuid.encode(u)
    >>> s
    'MLpZDiEXM4VsUryR9oE8uc'

    >>> shortuuid.decode(s) == u
    True

    >>> short = s[:7]
    >>> short
    'MLpZDiE'

    >>> h = shortuuid.decode(short)
    UUID('00000000-0000-0000-0000-009a5b27f8b9')

    >>> shortuuid.decode(shortuuid.encode(h)) == h
    True
    Class-based usage
    If you need to have various alphabets per-thread, you can use the ShortUUID class, like so:

    >>> su = shortuuid.ShortUUID(alphabet="01345678")
    >>> su.uuid()
    '034636353306816784480643806546503818874456'

    >>> su.get_alphabet()
    '01345678'

    >>> su.set_alphabet("21345687654123456")
    >>> su.get_alphabet()
    '12345678'
    """

    def __init__(self, alphabet: str | None = None) -> None:
        if alphabet is None:
            alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

        self.set_alphabet(alphabet)

    @property
    def _length(self) -> int:
        """Return the necessary length to fit the entire UUID given the current alphabet."""
        return int(ceil(log(2**128, self._alpha_len)))

    def encode(self, uuid: _uu.UUID, pad_length: int | None = None) -> str:
        """
        Encode a UUID into a string (LSB first) according to the alphabet.

        If leftmost (MSB) bits are 0, the string might be shorter.
        """
        if not isinstance(uuid, _uu.UUID):
            raise ValueError("Input `uuid` must be a UUID object.")
        if pad_length is None:
            pad_length = self._length
        return int_to_string(uuid.int, self._alphabet, padding=pad_length)

    def decode(self, string: str, legacy: bool = False) -> _uu.UUID:
        """
        Decode a string according to the current alphabet into a UUID.

        Raises ValueError when encountering illegal characters or a too-long string.

        If string too short, fills leftmost (MSB) bits with 0.

        Pass `legacy=True` if your UUID was encoded with a ShortUUID version prior to
        1.0.0.
        """
        if not isinstance(string, str):
            raise ValueError("Input `string` must be a str.")
        if legacy:
            string = string[::-1]
        return _uu.UUID(int=string_to_int(string, self._alphabet))

    def uuid(self, name: str | None = None, pad_length: int | None = None) -> str:
        """
        Generate and return a UUID.

        If the name parameter is provided, set the namespace to the provided
        name and generate a UUID.
        """
        if pad_length is None:
            pad_length = self._length

        # If no name is given, generate a random UUID.
        if name is None:
            u = _uu.uuid4()
        elif name.lower().startswith(("http://", "https://")):
            u = _uu.uuid5(_uu.NAMESPACE_URL, name)
        else:
            u = _uu.uuid5(_uu.NAMESPACE_DNS, name)
        return self.encode(u, pad_length)

    def random(self, length: int | None = None) -> str:
        """Generate and return a cryptographically secure short random string of `length`."""
        if length is None:
            length = self._length

        random_num = int(binascii.b2a_hex(os.urandom(length)), 16)
        return int_to_string(random_num, self._alphabet, padding=length)[:length]

    def get_alphabet(self) -> str:
        """Return the current alphabet used for new UUIDs."""
        return "".join(self._alphabet)

    def set_alphabet(self, alphabet: str) -> None:
        """Set the alphabet to be used for new UUIDs."""
        # Turn the alphabet into a set and sort it to prevent duplicates
        # and ensure reproducibility.
        new_alphabet = list(sorted(set(alphabet)))
        if len(new_alphabet) > 1:
            self._alphabet = new_alphabet
            self._alpha_len = len(self._alphabet)
        else:
            raise ValueError("Alphabet with more than one unique symbols required.")

    def encoded_length(self, num_bytes: int = 16) -> int:
        """Return the string length of the shortened UUID."""
        factor = log(256) / log(self._alpha_len)
        return int(ceil(factor * num_bytes))
