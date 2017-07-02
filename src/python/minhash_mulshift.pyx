cimport cython
from libc.stdlib cimport malloc, free

def calculate_minhashes(unsigned long[:, ::1] edges, unsigned int[:, ::1] signatures,
    unsigned long[::1] hash_a, unsigned long[::1] hash_b, unsigned int num_hashes):
    """
    Calculate the star min hash values by generating hash values for each fan-star combination
    and keeping the minimum values.
    :param edges: the fan-star edges
    :param signatures: the minhash signatures matrix (stars*signatures)
    :param hash_a: hash params a
    :param hash_b: hash params b
    :param num_hashes: the number of hashes to calculate for each fan-star pairing
    """
    with nogil:
        _calculate_minhashes(edges, signatures, hash_a, hash_b, num_hashes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _calculate_minhashes(unsigned long[:, ::1] edges,
    unsigned int[:, ::1] signatures, unsigned long[::1] hash_a, unsigned long[::1] hash_b, int num_hashes) nogil:
    cdef unsigned int fan, star_index, fan_previous = 0
    cdef unsigned int edge_count = edges.shape[0]
    # TODO get working with dynamically allocated memory currently seg-faulting.
    cdef unsigned int fan_stars[200000]
    cdef Py_ssize_t count, i, j, fs_index = 0
    cdef unsigned int* hashes = <unsigned int*> malloc(sizeof(unsigned int) * num_hashes)

    try:
        # Extracting the first fan row removes a needless if statement later on
        fan_previous = edges[0, 0]
        # Process all the fan-star rows
        for count in range(edge_count):
            fan = edges[count, 0]
            star_index = edges[count, 1]

            # Add the observed star to the fan's star collection
            if fan == fan_previous:
                fs_index += 1
                fan_stars[fs_index] = star_index
            else:
                # Process fan
                # 1. Calculate fan hashes
                for j in range(num_hashes):
                    # Note: using a multiply-shift hashing scheme to avoid computationally expensive modular arithmetic.
                    # From Wikipedia:
                    # "The state of the art for hashing integers is the multiply-shift scheme described by
                    # Dietzfelbinger et al. in 1997.[5] By avoiding modular arithmetic, this method is much easier to implement
                    # and also runs significantly faster in practice (usually by at least a factor of four[6])."
                    # See http://en.wikipedia.org/wiki/Universal_hashing for more information.
                    # From the same wiki page the c code given is: (unsigned) (a*x+b) >> (w-M) w is machine word size in bits
                    # M is bin size in powers of 2. This only works in C the unbounded integer shift lefts
                    # will not work in Python.
                    hashes[j] = (hash_a[j] * fan_previous + hash_b[j]) >> 33 # np.round(64-np.log2(3000000000)) 3e9 is max twitter ID
                # 2. Update star hashes
                for i in range(fs_index + 1):
                    for j in range(num_hashes):
                        if hashes[j] < signatures[fan_stars[i], j]:
                          signatures[fan_stars[i], j] = hashes[j]
                # 3. Reset local state
                fan_previous = fan
                fan_stars[0] = star_index
                fs_index = 0

        # Process last fan
        # 1. Calculate fan hashes
        for j in range(num_hashes):
            hashes[j] = (hash_a[j] * fan_previous + hash_b[j]) >> 33
        # 2. Update star hashes
        for i in range(fs_index + 1):
            for j in range(num_hashes):
                if hashes[j] < signatures[fan_stars[i], j]:
                    signatures[fan_stars[i], j] = hashes[j]
    finally:
        # Ensure the dynamically allocated memory is released
        free(hashes)

