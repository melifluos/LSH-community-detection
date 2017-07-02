 # encoding: utf-8
# filename: fan_loop.pyx
cimport cython
from libc.stdlib cimport malloc, free

# The largest unsigned 32bit prime (max uint32 - 4)
cdef unsigned long prime = 4294967291

def calculate_minhashes(unsigned int[:, ::1] edges, unsigned int[:, ::1] signatures,
    unsigned int[::1] a, unsigned int[::1] b, unsigned int max_fan_star,
    unsigned int offset, unsigned int num_hashes):
    """
    Calculate the star min hash values by generating hash values for each fan-star combination
    and keeping the minimum values.
    :param edges:
    :param signatures: A numpy array of shape (num_stars, num_hashes) and dtype np.uint32 intialised to all be the maximum
    value of a 32 bit unsigned integer
    :param a:
    :param b:
    :param max_fan_star:
    :param offset:
    :param num_hashes:
    :return:
    """
    with nogil:
        _calculate_minhashes(edges, signatures, a, b, max_fan_star, offset, num_hashes)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _calculate_minhashes(unsigned int[:, ::1] edges,
    unsigned int[:, ::1] signatures, unsigned int[::1] a, unsigned int[::1] b, int max_fan_star, int offset, int num_hashes) nogil:
    cdef unsigned int fan, star_index, fan_previous = 0
    cdef unsigned int edge_count = edges.shape[0]
    # TODO: DS find out why the dynamically allocated fan_stars array is producing seg faults
    # then replace the hard coded fan_stars array with that one
    #cdef unsigned int* fan_stars = <unsigned int*> malloc(sizeof(unsigned int) * max_fan_star)
    cdef unsigned int fan_stars[40000]
    cdef Py_ssize_t count, i, j, fs_index = 0
    cdef Py_ssize_t offset_end = num_hashes + offset
    cdef unsigned int* hashes = <unsigned int*> malloc(sizeof(unsigned int) * num_hashes)

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
                hashes[j] = ((a[j] * fan_previous) + b[j]) % prime
            # 2. Update star hashes
            for i in range(fs_index+1): # [0, fs_index]
                for j in range(num_hashes):
                    if hashes[j] < signatures[fan_stars[i], j+offset]:
                        signatures[fan_stars[i], j+offset] = hashes[j]
            # 3. Reset local state
            fan_previous = fan
            fan_stars[0] = star_index
            fs_index = 0

    # Process last fan
    # 1. Calculate fan hashes
    for j in range(num_hashes):
        hashes[j] = ((a[j] * fan_previous) + b[j]) % prime
    # 2. Update star hashes
    for i in range(fs_index+1): # [0, fs_index]
        for j in range(num_hashes):
            if hashes[j] < signatures[fan_stars[i], j+offset]:
                signatures[fan_stars[i], j+offset] = hashes[j]

    free(hashes)

